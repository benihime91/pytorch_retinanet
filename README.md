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
- <a href="https://colab.research.google.com/github/benihime91/pytorch_retinanet/blob/master/demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  [demo.ipynb](https://github.com/benihime91/pytorch_retinanet/blob/master/demo.ipynb)

## Installing Dependencies :
  Ensure that [python>=3.6](https://www.python.org/) , [torch>=1.6.0](https://pytorch.org/), torchvision>=0.7.0 is installed .
   ```bash
   $ git clone https://github.com/benihime91/pytorch_retinanet.git
   $ cd pytorch_retinanet
   $ pip install -r requirements.txt
   ```
   Note: for `pytorch-lightning` versions >= 1.0.0 t training will fail .
   
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
    from omegaconf import OmegaConf, DictConfig
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer

    from model import RetinaNetModel
    
    # load in the hparams yaml file
    hparams = OmegaConf.load("hparams.yaml")

    # instantiate lightning module
    model = RetinaNetModel(hparams)
    
    # Instantiate Trainer
    trainer = Trainer()
    # start train
    trainer.fit(model)
    # to test model using COCO API
    trainer.test(model)
   ```

## Loading Data:

The data can be loaded into the model in one of 3 ways.  

This is controlled via the `dataset.kind` parameter in `hparams.yaml`. 

1. To load in the in the COCO-2017 dataset: 
   * set `dataset.kind` = "coco"
   * set `dataset.root_dir` = {path to the coco dataset}
  
   ```yaml
   dataset:
      kind: coco
      root_dir: /Datasets/coco/
   ```

2. If the dataset is in Pascal-VOC format :
   * set  `dataset.kind` = "pascal"
   * set  `data.trn_paths` = [path_to_annotations, path_to_images]
   * similarly set the paths for valiation and test datasets

   ```yaml
   dataset:
      kind: pascal
      trn_paths:
         - /content/data_train/Annotations/
         - /content/data_train/Images/"
      test_paths:
         - /content/data_test/Annotations/"
         - /content/data_test/Images/"
      val_paths:
         - /content/data_validation/Annotations/"
         - /content/data_validation/Images/"
   ```
   Note: 
      * image and annotation folder can be the same folder.
      * *val_paths* if optional, if no validation data then set , 
  
      ```yaml
      val_paths: False
      ```

3. The datasets can also be loaded from a csv format. The csv file should be as follows :
   ```
   filename,width,height,class,xmin,ymin,xmax,ymax,labels
   Images/007826.jpg,500,375,diningtable,80,217,320,273,11
   Images/007826.jpg,500,375,chair,197,193,257,326,9
   ...
   ...
   Images/006286.jpg,500,375,diningtable,402,219,500,375,11
   Images/006286.jpg,500,375,diningtable,347,177,405,216,11
   ```
   * **`filename`** : path to the Image
   * **`width`** : width of the Image [Optional] 
   * **`height`** : height of the Image [Optional] 
   * **`class`** : class label for the particular annotation
   * **`labels`** : integer labels for the particular annotation
   * **`xmin`**, **`ymin`**, **`xmax`**, **`ymax`**: absolute bounding-box co-ordinates
  
   ```yaml
   dataset:
      kind: csv
      trn_paths: "train_data.csv"
      val_paths: "val_data.csv" #This is Optional
      test_paths: "test_data.csv" 
   ```

### Note : 
   - if validation dataset is not present set *hparams.dataset.val_paths = False.*
   - the model computes the COCO-API evaluation metrics on the test dataset.
   - for csv dataset each entry in the csv file should corresponding to a unique bounding-box.
   - labels should start from 1 as the 0th label is reserved for "__background__" class.
   - to generate a LABEL_MAP to be used for visulazation purposes:

   ```python
    from utils.pascal import generate_pascal_category_names
    import pandas as pd
    
    path = ... # path to csv file contraining the annotations
    df = pd.read_csv(path)
    
    # Generate a label map
    LABEL_MAP = generate_pascal_category_names(df) 
   ```
    

## Visualizing the bounding-box(s) over the image :

   ```python
   from utils import visualize_boxes_and_labels_on_image_array as vis_bbs
   from PIL import Image
   import cv2
   import numpy as np

   image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
   
   # or :
   # image = Image.open(image_path)
   # image = np.array(image)

   # normalize the image
   image = image / 255.0

   # label map should be a list containing contraining the name
   # of the categories. Each cotegory should be at the index 
   # corresponding to the integer category
   # 0th index is reserved for the background class
   LABEL_MAP = ...

   # (N,4) dimensional array containing the absolute bounding-box
   # co-ordinates in xmin, ymin, xmax, ymax, format. 
   boxes = ...
   # (N) dimensional array contraining the interger labels
   labels = ...
   # (N) dimensional array contraining the confidence probability for the image. 
   # This can also be None.
   scores = ...

   # draw bounding-box over the loaded image
   im = vis_bbs(image, boxes, labels, scores, LABEL_MAP)
   # this function returns a PIL image instance 
   # to view the image
   im.show()
   # or in jupyter-notebooks use : im
   
   ```

## References : 
```
https://arxiv.org/abs/1708.02002
https://github.com/facebookresearch/detectron2
https://github.com/pytorch/vision
https://github.com/fastai/course-v3/blob/master/nbs/dl2/pascal.ipynb
https://github.com/tensorflow/models/tree/master/research/object_detection
https://github.com/PyTorchLightning/wheat/tree/dee605b0bf5cf6b0ab08755c45e38dc07d338bb7
```
