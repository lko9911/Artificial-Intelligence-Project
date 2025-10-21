# model.py
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLabv3Plus_Depth(nn.Module):
    def __init__(self, output_channels=1):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=None)
        # classifier 마지막 layer 변경
        self.model.classifier[-1] = nn.Sequential(
            nn.Conv2d(256, output_channels, kernel_size=1),
            nn.ReLU()  # 음수 방지
        )

    def forward(self, x):
        return self.model(x)["out"]
