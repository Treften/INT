
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers=None):
        super().__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers or []

    def forward(self, data):
        return self.submodule(data)


class CNNNet(nn.Module):
    def __init__(self, num_classes, depth, data_size, emb_name=None,
                 pretrain_weight=None):
        super().__init__()

        if depth == 18:
            net = resnet18(weights=None)
        elif depth == 34:
            net = resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        net.fc = nn.Linear(net.fc.in_features, num_classes)

        if pretrain_weight:
            checkpoint = torch.load(pretrain_weight, map_location="cpu",
                                    weights_only=False)
            if isinstance(checkpoint, dict):
                checkpoint = checkpoint.get("state_dict",
                                            checkpoint.get("model_state_dict", checkpoint))
            checkpoint = {k.removeprefix("module."): v
                          for k, v in checkpoint.items()}
            missing, unexpected = net.load_state_dict(checkpoint, strict=False)
            print("Missing pretrained keys:", missing)
            print("Unexpected pretrained keys:", unexpected)

        self.net = net

    def forward(self, data):
        if data.ndim == 5:
            data = data.permute(0, 2, 4, 1, 3)

            if data.shape[1] == 1:
                data = data.squeeze(1)
            else:
                raise ValueError(
                    f"ResNet expects one frame, got depth={data.shape[1]}"
                )

        elif data.ndim == 4 and data.shape[-1] in (1, 3):
            data = data.permute(0, 3, 1, 2)

        return self.net(data)
