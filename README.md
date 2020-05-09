# Locally Adaptive Channel Attention-based Network for Denoising Images (IEEE ACCESS)
[![AUR](https://img.shields.io/aur/license/yaourt.svg?style=plastic)](LICENSE)

[Paper](https://ieeexplore.ieee.org/abstract/document/8999518) | [Bibtex](#Bibtex)


## Dependancy
```
pytorch >= 1.3
torchvision >= 0.4.1.
tensorboardX
hdf5
numpy
opencv
```

## Network Architecture
![graph](./images/Architecture.pdf)

## Train 
```
$ python train.py --preprocess True
```

## Test
```
$ python test.py --test_data Set12 (or Set68, Urban100) --output_size 10
```

## Bibtex
```
@article{lee2020locally,
  title={Locally Adaptive Channel Attention-Based Network for Denoising Images},
  author={Lee, Haeyun and Cho, Sunghyun},
  journal={IEEE Access},
  volume={8},
  pages={34686--34695},
  year={2020},
  publisher={IEEE}
}
```