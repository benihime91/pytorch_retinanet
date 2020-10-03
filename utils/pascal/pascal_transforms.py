import albumentations as A
from albumentations.pytorch import ToTensorV2


def compose_transforms(transforms=None):
    if transforms is None:
        transforms = []
    transforms.append(A.ToFloat(max_value=255.0, always_apply=True),)
    transforms.append(ToTensorV2(always_apply=True))

    c = A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    return c

