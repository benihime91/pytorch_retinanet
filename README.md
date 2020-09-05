# pytorch_retinanet
A simple implementations of `Retinanet` for `object detection` as described in the paper [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

## TODO: 
- [x] Create `Anchor Generator`.
- [x] Create `ResNet` based `BackBone Model`.
- [x] Create `FeaturePyramid` architecture as described in https://arxiv.org/abs/1612.03144.
- [x] [Focal Loss](https://arxiv.org/abs/1708.02002) & `Regeression` and `Classification` Head for `Retinanet`.
- [x] Assemble Retinanet Using `BackBone` => `FeaturePyramid` => `AnchorGenerator` => `Regeression` & `Classification` Head.
- [x] Decode `Retinanet Predictions`. 
- [x] Test model training on toy dataset.
  - [x] Check `Training`. [Failing: in losses.py:`cuda error`]
  - [x] Fix `cuda` errors in losses.py
  - [x] Evaluation using `COCO API`. [scripts taken from: https://github.com/PyTorchLightning/wheat]
  - [x] Fix `nan` losses during training.
- [x] Train on Pascal VOC 2007.
  - [x] Fix augmentation pipeline
  - [x] Log wandb & TensorBoard
  - [x] Evaluation using `COCO API`.
- [ ] Train on COCO dataset.

## Updates
- Started Training on [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) ran to many errors. 😭
- Fixed the errors & was able to train on the [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
- Check https://github.com/benihime91/retinanet_pet_detector.
- 27 August 2020:
  - Started training on [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html).
  - Ran into errors mainly due to `albumentations` removing boxes.
  - Fixed my image transformation pipeline.
  - Using PyTorchLightning to automate my training process.
