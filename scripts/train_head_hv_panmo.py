"""
Training script for sparse training, with only loss of the ground truth labels
"""

import os
import sys
sys.path.append('./')
import numpy as np
import joblib
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import nn

from torch.utils.tensorboard import SummaryWriter
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
from datasets.dataset import CoNICDatasetPanMon
import warnings
warnings.filterwarnings("ignore")

import time
from utils.util_funcs import setup, cleanup, run_function, rm_n_mkdir
from models.model_head_aug import HoVerNetHeadExt
from backbones.losses import SoftBCEWithLogitsLoss, MSGELoss, DiceLoss, FocalLoss, MSELoss, SoftCrossEntropyLoss

parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument('--focal_loss', default=False, action='store_true')
parser.add_argument('--scheduler', default=True, action='store_true')
parser.add_argument('--split', type=int, default=0)
parser.add_argument("--name", type=str, default="hover_paper_pannuke_seresnext50", help="name of experiment")
parser.add_argument("--encoder_name", type=str, default="seresnext50", help="name of the encoder")
parser.add_argument("--max_epoch", type=int, default=50, help="max epoch number")
parser.add_argument('--port', type=str, default="49152", help="port of the distributed training")
parser.add_argument("--augments", type=str, default="aug/aug_0.txt", help="augment file name")
parser.add_argument("--run", type = int, default = 0, help ="run number")
parser.add_argument('--pretrained_path', type=str, 
                default="/cluster/projects/nn12036k/tirilktr/pretrained/se_resnext50_32x4d-a260b3a4.pth", 
                # default="/cluster/projects/nn12036k/tirilktr/pretrained/se_resnext101_32x4d-3b2fe3d8.pth",
                help="pretrained weights")
args = parser.parse_args()
print(f"args: {args}")

def demo_basic(rank, world_size):
    """
    the base function to train with sparse labelling
    :param rank: rank of the thread
    :param world_size: total number of threads
    """
    setup(rank, world_size, args.port)

    if "pannuke" in args.name:
        print("Pannuke detected")
        # Load images, perform augmentation, generate horizontal and vertical maps
        imagenet = CoNICDatasetPanMon(img_path=f"/cluster/projects/nn12036k/tirilktr/datasets/pannuke/split_{args.split}/images_train.npy", 
                                                ann_path=f"/cluster/projects/nn12036k/tirilktr/datasets/pannuke/split_{args.split}/labels_train.npy",
                                      input_shape=(256, 256), mask_shape=(256, 256))
        num_types = 6
    elif "monusac" in args.name:
        print("Monusac detected")
        # Load images, perform augmentation, generate horizontal and vertical maps
        #imagenet = CoNICDatasetPanMon(img_path="/cluster/projects/nn12036k/tirilktr/datasets/monuseg/tissue_images_npy/train_images.npy", ann_path="data_monusac/labels_train.npy",
                                    #input_shape=(256, 256), mask_shape=(256, 256))
        #num_types = 5
    
    if rank == 0:
        print("RANK==0")
        print(args.name)
        #rm_n_mkdir("logs/{}".format(args.name))
        #writer = SummaryWriter("logs/{}".format(args.name))
        log_dir = f"logs/{args.name}/{args.encoder_name}/split_{args.split}/bs{args.batch_size}_epochs{args.max_epoch}_lr{args.lr}/run_{args.run}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print("len of training set: ", len(imagenet))
    sampler = DistributedSampler(imagenet)
    print("Sampler loaded")
    imagenet_dataloader = torch.utils.data.DataLoader(dataset=imagenet, batch_size=args.batch_size,
                                                      num_workers=1, sampler=sampler)
    
    model = HoVerNetHeadExt(num_types=num_types, freeze=False, pretrained_backbone=args.pretrained_path, encoder_name=args.encoder_name)
    # model = create_model()
    model = model.to(rank)
    ddp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(ddp_model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(ddp_model.module.parameters(), lr=args.lr)
    milestones = list(range(40, args.max_epoch, 10)) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    soft_bce_loss = SoftBCEWithLogitsLoss()

    ce_loss_tp = torch.nn.CrossEntropyLoss()
    msge_loss = MSGELoss()
    dice_loss = DiceLoss(mode="multiclass")
    mse_loss = MSELoss()
    focal_loss = FocalLoss(mode="multiclass")

    dist.barrier()

    for epoch_idx in range(args.max_epoch):
        ddp_model.train()
        loss_sum = 0
        loss_np_sum = 0
        loss_hv_sum = 0
        loss_tp_sum = 0
        loss_focal_sum = 0
        sampler.set_epoch(epoch_idx)

        loss = 0

        if rank == 0:
            start_time = time.time()
        
        for image, np_map, hv_map, tp_map in imagenet_dataloader:
            image = image.to(rank).float()
            np_map = np_map.to(rank).float()
            hv_map = hv_map.to(rank).float()
            tp_map = tp_map.to(rank).float()

            np_predicted, hv_predicted, tp_predicted = ddp_model(image)

            hv_predicted = torch.tanh(hv_predicted)

            loss_np = 2 * soft_bce_loss(np_predicted, np_map) + 2 * dice_loss(np_predicted, np_map)
            loss_hv = 2 * mse_loss(hv_predicted, hv_map) + 2* msge_loss(hv_predicted, hv_map, np_map, device=rank)

            loss_tp = soft_bce_loss(tp_predicted, tp_map)  + dice_loss(tp_predicted, tp_map)

            # add a weighted cross entropy loss to the type branch
            tp_map = torch.argmax(tp_map, dim=1)

            tp_predicted = tp_predicted.permute(0, 2, 3, 1).contiguous()
            tp_predicted = tp_predicted.view(-1, tp_predicted.shape[-1])
            tp_map = tp_map.view(-1)
            loss_tp += 2 * ce_loss_tp(tp_predicted, tp_map)

            loss = loss_np + loss_hv + loss_tp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                loss_sum += loss.detach().cpu().numpy()
                loss_np_sum += loss_np.detach().cpu().numpy()
                loss_hv_sum += loss_hv.detach().cpu().numpy()
                loss_tp_sum += loss_tp.detach().cpu().numpy()

        if args.scheduler: 
            scheduler.step()
    
        if rank == 0:
            print("loss \t loss_np_sum \t loss_hv_sum \t loss_tp_sum \t loss_focal_sum")

            if args.focal_loss:
                print(f"{loss_sum} \t {loss_np_sum} \t {loss_hv_sum} \t {loss_tp_sum} \t {loss_focal_sum}")
            else:
                print(f"{loss_sum} \t {loss_np_sum} \t {loss_hv_sum} \t {loss_tp_sum}")

            print(f"========================epoch {epoch_idx}, time: {time.time()-start_time}s============")
            writer.add_scalar('Loss/loss', loss_sum / len(imagenet), epoch_idx)
            writer.add_scalar('Loss/loss_np_sum', loss_np_sum / len(imagenet), epoch_idx)
            writer.add_scalar('Loss/loss_hv_sum', loss_hv_sum / len(imagenet), epoch_idx)
            writer.add_scalar('Loss/loss_tp_sum', loss_tp_sum / len(imagenet), epoch_idx)
            if args.focal_loss:
                writer.add_scalar('Loss/loss_focal_sum', loss_focal_sum / len(imagenet), epoch_idx)

            # check point save   
            # os.makedirs("/cluster/projects/nn12036k/tirilktr/pannuke_output/checkpoints/{}/bs{}_epochs{}_lr{}/run_{}/".format(args.encoder_name, args.batch_size, args.max_epoch, args.lr, args.run), exist_ok=True)
            # if epoch_idx > 30:
            #     checkpoint_path = "/cluster/projects/nn12036k/tirilktr/pannuke_output/checkpoints/seresnext50/bs{}_epochs{}_lr{}/run_{}/improved-net_{}.pt".format(args.run, args.batch_size, args.max_epoch, args.lr, epoch_idx)
            #     torch.save(ddp_model.module.state_dict(), checkpoint_path)
            # checkpoint_path = "/cluster/projects/nn12036k/tirilktr/pannuke_output/checkpoints/seresnext50/bs{}_epochs{}_lr{}/run_{}/improved-net_latest.pt".format(args.run, args.batch_size, args.max_epoch, args.lr)
            # torch.save(ddp_model.module.state_dict(), checkpoint_path)

            ckpt_dir = f"/cluster/projects/nn12036k/tirilktr/pannuke_output/checkpoints/{args.encoder_name}/split_{args.split}/bs{args.batch_size}_epochs{args.max_epoch}_lr{args.lr}/run_{args.run}"
            os.makedirs(ckpt_dir, exist_ok=True)
            if epoch_idx > 50:
                torch.save(ddp_model.module.state_dict(), f"{ckpt_dir}/improved-net_{epoch_idx}.pt")
            torch.save(ddp_model.module.state_dict(), f"{ckpt_dir}/improved-net_latest.pt")


        dist.barrier()

    if rank == 0:
        writer.close()

    cleanup()


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"  
    n_gpus = torch.cuda.device_count()
    x = torch.cuda.is_available()
    print(f"n_gpus: {n_gpus}")
    print(x)
    run_function(demo_basic, n_gpus)
