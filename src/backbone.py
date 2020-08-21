from typing import *
import torch.nn as nn
import torchvision
from torch import nn as nn
from torch.functional import Tensor


__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

loaders = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
}

# Dictionary to store Intermediate Outputs
inter_outs = {}


def hook_outputs(self, inp, out) -> None:
    inter_outs[self] = out


class EmptyLayer(nn.Module):
    " PlaceHolder for `AvgPool` and `FC Layer` "

    def __init__(self) -> None:
        super(EmptyLayer, self).__init__()

    def forward(self, xb):
        return xb


class BackBone(nn.Module):
    def __init__(
        self,
        kind: str = "resnet18",
        hook_fn: Callable = None,
        pretrained: bool = True,
        freeze_bn: bool = True,
    ) -> None:
        """Create a Backbone from `kind`"""
        super(BackBone, self).__init__()
        build_fn = loaders[kind]
        self.backbone = build_fn(pretrained=pretrained)
        self.backbone.avgpool = EmptyLayer()
        self.backbone.fc = EmptyLayer()

        self.backbone.layer2.register_forward_hook(hook_fn)
        self.backbone.layer3.register_forward_hook(hook_fn)
        self.backbone.layer4.register_forward_hook(hook_fn)

        # Freeze batch_norm: Not sure why ?? but every other implementation does it
        if freeze_bn:
            for layer in self.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()

    def forward(self, xb: Tensor) -> List[Tensor]:
        _ = self.backbone(xb)
        out = [
            inter_outs[self.backbone.layer2],
            inter_outs[self.backbone.layer3],
            inter_outs[self.backbone.layer4],
        ]
        return out


def get_backbone(
    kind: str = "resnet18", pretrained: bool = True, freeze_bn: bool = True
) -> nn.Module:
    """
    Returns a `ResNet` Backbone.

    Args:
        1. kind       : (str) name of the resnet model eg: `resnet18`.
        2. pretrained : (bool) wether to load pretrained `imagenet` weights.
        3. freeze_bn  : (bool) wether to freeze `BatchNorm` layers.

    Example:
        >>> m = get_backbone(kind='resnet18')
    """
    assert kind in __all__, f"`kind` must be one of {__all__} got {kind}"
    backbone = BackBone(
        kind=kind, hook_fn=hook_outputs, pretrained=pretrained, freeze_bn=freeze_bn
    )
    return backbone


# if __name__ == '__main__':
#     import torch
#     m = get_backbone()
#     z = torch.rand([1, 3, 64, 64])
#     output = m(z)
#     print(output)
#     print([o.shape for o in output])
