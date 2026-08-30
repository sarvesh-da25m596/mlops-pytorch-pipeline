"""Model factory. Only resnet18 is supported for now (assignment scope)."""
import torch.nn as nn
from torchvision.models import resnet18


def get_model(architecture: str, num_classes: int = 10) -> nn.Module:
    if architecture != "resnet18":
        raise ValueError(f"unsupported architecture: {architecture}")

    model = resnet18(weights=None, num_classes=num_classes)
    # stock stem (7x7 conv + maxpool) is tuned for 224x224 ImageNet images and
    # downsamples 32x32 CIFAR inputs too aggressively -> swap for a 3x3 conv,
    # drop the maxpool. Standard trick for training resnets on CIFAR-10.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
