
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class StatisticsPooling(nn.Module):
    def forward(self, x):
        # x: [B, C, F, T]
        mean = x.mean(dim=-1)
        var = x.var(dim=-1, correction=0)

        # Агрегируем частотную ось после статистик по времени.
        mean = mean.mean(dim=-1)  # [B, C]
        var = var.mean(dim=-1)    # [B, C]

        # [B, 2*C]
        return torch.cat([mean, var], dim=1)


class STPNet(nn.Module):
    def __init__(self, num_classes, depth):
        super().__init__()

        resnet_builders = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        if depth not in resnet_builders:
            raise ValueError(
                f"Unsupported ResNet depth: {depth}. "
                f"Choose one of {list(resnet_builders)}."
            )

        # Без предобученных весов: классификационная задача другая.
        backbone = resnet_builders[depth](weights=None)

        # Спектрограмма имеет один канал вместо трёх RGB.
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Извлекаем карты признаков, не используя avgpool и fc ResNet.
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        channels = backbone.fc.in_features

        self.pool = StatisticsPooling()

        self.net = nn.Sequential(
            self.features,
            # [B, C, F, T] -> [B, 2*C]
            self.pool,
            nn.Linear(2 * channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, data):
        # Ожидается [B, 1, 80, T].
        # Если подаётся [B, 80, T], добавляем канал.
        if data.ndim == 3:
            data = data.unsqueeze(1)

        if data.ndim != 4:
            raise ValueError(
                f"Expected [B, 1, F, T] or [B, F, T], "
                f"got {tuple(data.shape)}"
            )

        output = self.net(data)
        return output
