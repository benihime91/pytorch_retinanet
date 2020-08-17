from pytorch_retinanet.anchors import AnchorGenerator
from pytorch_retinanet.backbone import get_backbone
from pytorch_retinanet.modelling import Retinanet
import torch

def test():
    model = get_backbone()
    outputs = model(torch.ones((1, 3, 600, 600)))
    ag = AnchorGenerator()
    anchors = ag(outputs)
    # print(anchors)
    print([o.shape for o in outputs])
    print([a.shape for a in anchors])

if __name__ == "__main__":
    test()
