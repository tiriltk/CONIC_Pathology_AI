import os
import sys
sys.path.append("./")
import cv2

import numpy as np
# from utils.visualize_gt import draw_dilation_seg

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def pannuke(split):
    print("========================================")
    print(f"Processing fold {split}")

    images_train = []
    labels_train = []

    print("Loading data")

    images_this = np.load(f"/Users/tirilkt/Documents/studie/masteroppgave/Datasets/Pannuke/Fold_{split+1}/images/images.npy", mmap_mode='r')
    masks_this  = np.load(f"/Users/tirilkt/Documents/studie/masteroppgave/Datasets/Pannuke/Fold_{split+1}/masks/masks.npy",  mmap_mode='r')

    print("Loaded data, starting iteration")

    for img_single, mask_single in zip(images_this, masks_this):
        images_train.append(img_single)

        inst_map = np.zeros((256, 256))
        num_nuc = 0
        for j in range(5):
            # copy value from new array if value is not equal 0
            layer_res = remap_label(mask_single[:, :, j])
            # inst_map = np.where(mask[:,:,j] != 0, mask[:,:,j], inst_map)
            inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
            num_nuc = num_nuc + np.max(layer_res)
        inst_map = remap_label(inst_map)

        type_map = np.zeros((256, 256)).astype(np.int32)
        for j in range(5):
            layer_res = ((j + 1) * np.clip(mask_single[:, :, j], 0, 1)).astype(np.int32)
            type_map = np.where(layer_res != 0, layer_res, type_map)
                
        inst_type = np.stack((inst_map, type_map), axis=-1)
        labels_train.append(inst_type)


    images_train = np.array(images_train)
    labels_train = np.array(labels_train)
    print("images_train shape: ", images_train.shape)
    print("labels_train shape: ", labels_train.shape)

    return images_train, labels_train


#select the folds for training and test
def select_folds(test_fold: int):
    #return two training folds and one test fold
    folds = [0,1,2] #folds to choose from
    train_folds = []
    for fold in folds:
        if fold != test_fold:
            train_folds.append(fold)
    return train_folds, test_fold

#builds the combined dataset, with two folds for training and one for test
def pannuke_combined(test_fold=0):
    train_folds, test_fold = select_folds(test_fold)
    print(f"Train folds: {train_folds} Test fold: {test_fold}")

    #training, two folders
    train_images_list, train_labels_list = [], []
    for f in train_folds:
        images, labels = pannuke(f)
        train_images_list.append(images)
        train_labels_list.append(labels)
    train_images = np.concatenate(train_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    #test, one fold
    test_images, test_labels = pannuke(test_fold)

    #saving
    out_dir = f"/Users/tirilkt/Documents/studie/masteroppgave/Datasets/Pannuke/pannuke_datasets_combined/split_{test_fold}/"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "images_train.npy"), train_images)
    np.save(os.path.join(out_dir, "labels_train.npy"), train_labels)
    np.save(os.path.join(out_dir, "images_test.npy"),  test_images)
    np.save(os.path.join(out_dir, "labels_test.npy"),  test_labels)
    print(f"Saved combined data to {out_dir}")

if __name__ == "__main__":
    pannuke_combined(test_fold=0)   #Train: Fold 1 + 2, Test: Fold 0
    # pannuke_combined(test_fold=1) #Train: Fold 0 + 2 Test: Fold 1
    # pannuke_combined(test_fold=2) #Train: Fold 0 + 1, Test: Fold 2


"""

def save_train(train_folds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_images_list, train_labels_list = [], []

    for f in train_folds:
        imgs, lbls = pannuke(f)
        train_images_list.append(imgs)
        train_labels_list.append(lbls)
       
    train_images = np.concatenate(train_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    np.save(os.path.join(out_dir, "images_train.npy"), train_images)
    np.save(os.path.join(out_dir, "labels_train.npy"), train_labels)


def save_test(test_fold, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    test_images, test_labels = pannuke(test_fold)

    np.save(os.path.join(out_dir, "images_test.npy"),  test_images)
    np.save(os.path.join(out_dir, "labels_test.npy"),  test_labels)


def pannuke_combined(test_fold=0):
    train_folds = [f for f in [0,1,2] if f != test_fold]
    out_dir = f"/Users/tirilkt/Documents/studie/masteroppgave/Datasets/Pannuke/pannuke_datasets_combined/split_{test_fold}/"
    print(f"Train folds: {train_folds}, Test fold: {test_fold}")

    save_train(train_folds, out_dir) 
    save_test(test_fold, out_dir)   

if __name__ == "__main__":
    test_fold = 0
    out_dir = "/Users/tirilkt/Documents/studie/masteroppgave/Datasets/Pannuke/pannuke_datasets_combined/split_0/"
    train_folds, _ = select_folds(test_fold)
    save_train(train_folds, out_dir)
    print("Saving to:", out_dir)

    # pannuke_combined(test_fold=0)   # use 0 as test, then folds 1+2 train
    # pannuke_combined(test_fold=1)
    # pannuke_combined(test_fold=2)

"""