from pytorch_retinanet.models import Retinanet
import warnings
import torch
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    model = Retinanet(num_classes=8, backbone_kind='resnet18', pretrained=False)
    z = torch.rand([3, 3, 600, 600])
    model.eval()
    outputs = model(z)
    print(f" --> OUTPUTS:\n{outputs}")
