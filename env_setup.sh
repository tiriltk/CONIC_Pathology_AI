cd ../HoVerNet_baseline/
conda env create -f environment.yml
conda activate hovernet-baseline
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch
pip install albumentations>=0.3.2 --no-binary imgaug,albumentations
pip install opencv-python==4.5.5.64
pip install pretrainedmodels
pip install efficientnet_pytorch
pip install timm==0.6.13