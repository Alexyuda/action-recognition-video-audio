#action recofnition using audio and video

This project is based on the the following repos:
https://github.com/piergiaj/pytorch-i3d
https://github.com/keunhong/pytorch-soundnet

## project structure

## installation 
1. Download soundnet from https://github.com/keunhong/pytorch-soundnet and rename to soundnet.
2. Download i3d from https://github.com/piergiaj/pytorch-i3d and rename to i3d.
3. Download ucf101 videos and train/test splits from https://www.crcv.ucf.edu/data/UCF101.php
5. Resize, crop and remove videos without sound:
python data_processing.py --input_dir {ucf location dir} --output_dir {scales and crop version of ucf}
6. Edit ops.py


## train


## References
SoundNet: Learning Sound Representations from Unlabeled Video
Yusuf- Carl- Torralba- Antonio - https://arxiv.org/abs/1610.09001

Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
Carreira- Joao- Zisserman- Andrew - https://arxiv.org/abs/1705.07750

UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild
Soomro- Khurram- Zamir-Amir Roshan- Shah- Mubarak - https://arxiv.org/abs/1212.0402


