
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class CNN3D(nn.Module):
    def __init__(self, in_channels=3, output_size=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Linear(128, output_size)
        
    def forward(self, x):
        #print("3D input:", x.shape)
        x = self.features(x)
        #print("3D features:", x.shape)
        x = x.mean(dim=(-1, -2)).transpose(1, 2)
        x = self.head(x)
        #print("3D output:", x.shape)
        return x.reshape(-1, x.shape[-1])


# class ResidualBlock3D(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=0.0):
#         super().__init__()

#         self.conv1 = nn.Conv3d(
#             in_channels, out_channels, kernel_size=3,
#             padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm3d(out_channels)

#         self.conv2 = nn.Conv3d(
#             out_channels, out_channels, kernel_size=3,
#             padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm3d(out_channels)

#         self.shortcut = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm3d(out_channels)
#         ) if in_channels != out_channels else nn.Identity()

#         self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         identity = self.shortcut(x)

#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out = self.dropout(out)

#         return self.relu(out + identity)


# class CNN3D(nn.Module):
#     def __init__(self, in_channels=3, output_size=2, dropout=0.2):
#         super().__init__()

#         self.features = nn.Sequential(
#             ResidualBlock3D(in_channels, 32),
#             nn.MaxPool3d((1, 2, 2)),

#             ResidualBlock3D(32, 64),
#             nn.MaxPool3d((1, 2, 2)),

#             ResidualBlock3D(64, 128, dropout=dropout),
#             nn.MaxPool3d((1, 2, 2)),

#             ResidualBlock3D(128, 128, dropout=dropout),
#         )

#         self.head = nn.Linear(128, output_size)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.mean(dim=(-1, -2)).transpose(1, 2)
#         x = self.head(x)
#         return x.reshape(-1, x.shape[-1])


class CNNNet(nn.Module):
    def __init__(self, num_classes, depth, data_size, emb_name=None,
                 pretrain_weight=None, mode="ResNet", in_channels=3):
        super().__init__()

        self.mode = mode

        if mode == "CNN3D":
            self.net = CNN3D(
                in_channels=in_channels,
                output_size=num_classes,
            )

        elif mode == "ResNet":
            if depth == 18:
                net = resnet18(weights=None)
            elif depth == 34:
                net = resnet34(weights=None)
            else:
                raise ValueError(f"Unsupported ResNet depth: {depth}")

            # if depth == 18:
            #     weights = None if pretrain_weight else ResNet18_Weights.DEFAULT
            #     net = resnet18(weights=weights)

            # elif depth == 34:
            #     weights = None if pretrain_weight else ResNet34_Weights.DEFAULT
            #     net = resnet34(weights=weights)

            # else:
            #     raise ValueError(f"Unsupported ResNet depth: {depth}")

            net.fc = nn.Linear(net.fc.in_features, num_classes)
            # for param in net.parameters():
            #     param.requires_grad = False

            # for param in net.layer4.parameters():
            #     param.requires_grad = True

            # for param in net.fc.parameters():
            #     param.requires_grad = True
            if pretrain_weight:
                checkpoint = torch.load(
                    pretrain_weight,
                    map_location="cpu",
                    weights_only=False,
                )
                if isinstance(checkpoint, dict):
                    checkpoint = checkpoint.get(
                        "state_dict",
                        checkpoint.get("model_state_dict", checkpoint),
                    )

                checkpoint = {
                    k.removeprefix("module."): v
                    for k, v in checkpoint.items()
                }

                missing, unexpected = net.load_state_dict(
                    checkpoint, strict=False
                )
                print("Missing pretrained keys:", missing)
                print("Unexpected pretrained keys:", unexpected)

            self.net = net

        else:
            raise ValueError(f"Unsupported network type: {mode}")

    def forward(self, data):
        if data.ndim == 5:
            if self.mode == "CNN3D":
                # [B, H, D, W, C] -> [B, C, D, H, W]
                data = data.permute(0, 4, 2, 1, 3).contiguous()

            else:
                # [B, H, D, W, C] -> [B, D, C, H, W]
                data = data.permute(0, 2, 4, 1, 3).contiguous()

                if data.shape[1] != 1:
                    raise ValueError(
                        f"ResNet expects depth=1, got depth={data.shape[1]}"
                    )

                data = data.squeeze(1)

        elif data.ndim == 4:
            # [B, H, W, C] -> [B, C, H, W]
            data = data.permute(0, 3, 1, 2).contiguous()

        return self.net(data)
