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
   Clone the Github Repo
   ```bash
   $ git clone https://github.com/benihime91/pytorch_retinanet.git
   ```

   For easy training pipeline, we recommend using **`pytorch-lightning`** for **training** and **testing**.  
   
   
   First of all open the **`hparams.yaml`** file and modify it according to need. Instructions to 
   modeify the same are present inside the file.  
   
   
   Create a python script inside the `retinanet repo`. Name it whatever you want and then insert the 
   following lines:
   ```python
    from model import RetinaNetModel, LogCallback
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from omegaconf import OmegaConf, DictConfig

    # use random seed so that results are reproducible
    pl.seed_everything(123)
    
    # load in the hparams yaml file
    hparams = OmegaConf.load("hparams.yaml")

    # instantiate lightning module
    model = RetinaNetModel(hparams=hparams)

    # instantiate lightning-trainer for trian and test
    # Trainer specific arguments see 
    # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
    trainer = Trainer(precision=16, 
                      # if training on GPU
                      gpus=1, 
                      callbacks=[LogCallback()], 
                      weights_summary=None,
                      terminate_on_nan = True, 
                      deterministic=True,
                      # total number of epochs to train for
                      max_epochs=1, 
                      )

    # start train
    trainer.fit(model)

    # to test model using COCO API
    trainer.test(model)
   ```