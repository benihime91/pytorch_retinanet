# pytorch_retinanet
A `PyTorch` implementation of `Retinanet` for object detection as described in the paper **[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).**

The code is heavily influended by [Detectron2](https://github.com/facebookresearch/detectron2) , torchvision implementation of [RCNN models](https://github.com/pytorch/vision/tree/master/torchvision/models/detection) and the [FastAI implementation](https://github.com/fastai/course-v3/blob/master/nbs/dl2/pascal.ipynb)

## TODO: 
- [x] Create `Anchor Generator`.
- [x] Create `ResNet` based `BackBone Model`.
- [x] Create `FeaturePyramid` architecture as described in https://arxiv.org/abs/1612.03144.
- [x] [Focal Loss](https://arxiv.org/abs/1708.02002) & `Regeression` and `Classification` Head for `Retinanet`.
- [x] Assemble Retinanet Using `BackBone` => `FeaturePyramid` => `AnchorGenerator` => `Regeression` & `Classification`.

## Tutorials:

## Usage:
