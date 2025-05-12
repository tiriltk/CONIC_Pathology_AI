conda env create -f environment.yml
conda activate aughovernet
conda install pytorch==2.5.1 torchvision==0.20.1 cudatoolkit==11.8 -c pytorch
pip install albumentations>=2.0.4 --no-binary imgaug,albumentations
pip install opencv-python
pip install pretrainedmodels
pip install efficientnet_pytorch
pip install timm