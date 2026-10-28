
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch.common.losses import *


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers=None):
        super().__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers or []

    def forward(self, data):
        return self.submodule(data)


class CNNNet(nn.Module):
    def __init__(
        self,
        num_classes,
        depth,
        data_size,
        emb_name=None,
        pretrain_weight=None,
    ):
        super().__init__()

        in_channels = data_size.get("channels", data_size.get("depth", 3))
        width = data_size["width"]

        num_blocks = max(2, min(int(depth), 6))

        layers = []
        channels = [32, 64, 128, 256, 256, 512]

        for i in range(num_blocks):
            out_channels = channels[i]

            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])

            in_channels = out_channels

        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        ])

        self.net = nn.Sequential(*layers)

        if pretrain_weight is not None:
            checkpoint = torch.load(
                pretrain_weight,
                map_location="cpu",
                weights_only=True,
            )

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]

            checkpoint = {
                k.removeprefix("module."): v
                for k, v in checkpoint.items()
            }

            self.load_state_dict(checkpoint)

    def forward(self, data):
        return self.net(data)
