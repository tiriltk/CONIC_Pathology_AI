python scripts/train_head_hv_panmo.py --encoder_name seresnext50 --pretrained_path pretrained/se_resnext50_32x4d-a260b3a4.pth --name hover_paper_pannuke_seresnext50_split_0 --split 0
python scripts/train_head_hv_panmo.py --encoder_name seresnext101 --pretrained_path pretrained/se_resnext101_32x4d-3b2fe3d8.pth --name hover_paper_pannuke_seresnext101_split_0 --split 0

python scripts/train_head_hv_panmo.py --encoder_name seresnext50 --pretrained_path pretrained/se_resnext50_32x4d-a260b3a4.pth --name hover_paper_pannuke_seresnext50_split_1 --split 1
python scripts/train_head_hv_panmo.py --encoder_name seresnext101 --pretrained_path pretrained/se_resnext101_32x4d-3b2fe3d8.pth --name hover_paper_pannuke_seresnext101_split_1 --split 1


python scripts/train_head_hv_panmo.py --encoder_name seresnext50 --pretrained_path pretrained/se_resnext50_32x4d-a260b3a4pth --name hover_paper_pannuke_seresnext50_split_2 --split 2
python scripts/train_head_hv_panmo.py --encoder_name seresnext101 --pretrained_path pretrained/se_resnext101_32x4d-3b2fe3d8.pth --name hover_paper_pannuke_seresnext101_split_2 --split 2

python scripts/wenhua_eval_hv_panmo_ensemble_all.py --exp_name0 hover_paper_pannuke_seresnext50_split_0 --exp_name1 hover_paper_pannuke_seresnext101_split_0
python scripts/wenhua_eval_hv_panmo_ensemble_all.py --exp_name0 hover_paper_pannuke_seresnext50_split_1 --exp_name1 hover_paper_pannuke_seresnext101_split_1 --split 1
python scripts/wenhua_eval_hv_panmo_ensemble_all.py --exp_name0 hover_paper_pannuke_seresnext50_split_2 --exp_name1 hover_paper_pannuke_seresnext101_split_2 --split 2
