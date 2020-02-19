## Action recognition using audio and video on ucf101
This project is based on the the following repos:
https://github.com/piergiaj/pytorch-i3d
https://github.com/keunhong/pytorch-soundnet

### Installation 
1. Download soundnet from https://github.com/keunhong/pytorch-soundnet and rename to soundnet.
2. Download i3d from https://github.com/piergiaj/pytorch-i3d and rename to i3d.
3. Download ucf101 videos and train/test splits from https://www.crcv.ucf.edu/data/UCF101.php
5. Resize, crop and remove videos without sound:
python data_processing.py --input_dir {ucf location dir} --output_dir UCF-101-rescaled-cropped
6. Edit root_dir in opts.py:
https://github.com/Alexyuda/action_recognition/blob/fae5b5b6d826674d2d7f531a602b9c801ac63237/opts.py#L6
7. Download pretrained networks from:


### Train
python Main.py --model_type {choose i3d/i3d_soundnet_concat/i3d_soundnet_attention} --train_or_test_mode train

### Test
python Main.py --train_or_test_mode test --model_type {choose i3d/i3d_soundnet_concat/i3d_soundnet_attention} --use_pre_trained_model True --pre_trained_model_name {model name}

### References
SoundNet: Learning Sound Representations from Unlabeled Video
Yusuf- Carl- Torralba- Antonio - https://arxiv.org/abs/1610.09001

Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
Carreira- Joao- Zisserman- Andrew - https://arxiv.org/abs/1705.07750

UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild
Soomro- Khurram- Zamir-Amir Roshan- Shah- Mubarak - https://arxiv.org/abs/1212.0402


