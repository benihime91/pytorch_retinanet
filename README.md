# pytorch_retinanet
A simple implementations of `Retinanet` for `object detection` as described in the paper [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

> This project uses [PyTorch](https://pytorch.org/) and [PyTorchLightning](https://github.com/PyTorchLightning/pytorch-lightning).

## AIM: 
- Unlike other implementations make the code easy to read , understand & replicate.
- Easy DataLoading.

## TODO: 
- [x] Create `Anchor Generator`.
- [x] Create `ResNet` based `BackBone Model`.
- [x] Create `FPN` architecture as described in https://arxiv.org/abs/1612.03144.
- [ ] [Focal Loss](https://arxiv.org/abs/1708.02002) & `Regeression` and `Classification` Head for `Retinanet`.
- [ ] Assemble Retinanet Using `BackBone` => `FPN` => `AnchorGenerator` => `Regeression` & `Classification` Head.
- [ ] Decode `Retinanet Predictions`. 
- [ ] Create `Dataset` for Loading Custom Data. [Preferably from CSV].
- [ ] Training Loop using `PyTorch Lightning`.
- [ ] Inference for `single image` and `batches`.
- [ ] Scripts for `training` & `inference`.
- [ ] Add Docs.
- [ ] Evaluation using `COCO API`.
