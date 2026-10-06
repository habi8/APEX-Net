import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121


class APAM(nn.Module):
    """Anatomy Prior Attention Module for a feature map and spatial mask."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")

        hidden_channels = max(1, in_channels // reduction)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.clr1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(0.2),
        )
        self.clr2 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(0.2),
        )
        self.clr3 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(0.2),
        )
        self.clr4 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(0.2),
        )
        self.cs = nn.Sequential(
            nn.Linear(hidden_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, feature_map, mask):
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B, C, H, W]")
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must have shape [B, 1, H, W]")
        if feature_map.shape[0] != mask.shape[0]:
            raise ValueError("feature_map and mask batch sizes must match")
        if feature_map.shape[1] != self.clr1[0].in_features:
            raise ValueError(
                f"feature_map must have {self.clr1[0].in_features} channels"
            )

        mask = mask.to(device=feature_map.device, dtype=feature_map.dtype)
        if mask.shape[-2:] != feature_map.shape[-2:]:
            mask = F.interpolate(mask, size=feature_map.shape[-2:], mode="nearest")

        masked_features = feature_map * mask
        feature_avg = self.global_avg_pool(feature_map).flatten(1)
        feature_max = self.global_max_pool(feature_map).flatten(1)
        masked_avg = self.global_avg_pool(masked_features).flatten(1)
        masked_max = self.global_max_pool(masked_features).flatten(1)

        channel_weights = self.cs(
            self.clr1(feature_avg)
            + self.clr2(feature_max)
            + self.clr3(masked_avg)
            + self.clr4(masked_max)
        ).unsqueeze(-1).unsqueeze(-1)

        return channel_weights * feature_map + (1 - channel_weights) * masked_features


class APEXNet(nn.Module):
    """DenseNet-121 classifier using lung ROI and disease-prior attention.

    Args:
        x: RGB images with shape [B, 3, H, W].
        lung_mask: Lung ROI masks with shape [B, 1, H, W].
        disease_prior_maps: Disease-specific spatial maps with shape
            [B, num_classes, H, W], ordered to match the output classes.
    """

    def __init__(self, num_classes=14, pretrained=True, reduction=16):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        self.features = backbone.features
        feature_channels = backbone.classifier.in_features
        self.num_classes = num_classes
        self.lung_apams = nn.ModuleList(
            APAM(feature_channels, reduction=reduction) for _ in range(num_classes)
        )
        self.prior_apams = nn.ModuleList(
            APAM(feature_channels, reduction=reduction) for _ in range(num_classes)
        )
        self.classifiers = nn.ModuleList(
            nn.Linear(feature_channels * 2, 1) for _ in range(num_classes)
        )

    def forward(self, x, lung_mask, disease_prior_maps):
        if x.ndim != 4:
            raise ValueError("x must have shape [B, 3, H, W]")
        if lung_mask.ndim != 4 or lung_mask.shape[1] != 1:
            raise ValueError("lung_mask must have shape [B, 1, H, W]")
        if disease_prior_maps.ndim != 4:
            raise ValueError(
                "disease_prior_maps must have shape [B, num_classes, H, W]"
            )
        if lung_mask.shape[0] != x.shape[0]:
            raise ValueError("x and lung_mask batch sizes must match")
        if disease_prior_maps.shape[:2] != (x.shape[0], self.num_classes):
            raise ValueError(
                "disease_prior_maps must have shape "
                f"[{x.shape[0]}, {self.num_classes}, H, W]"
            )

        feature_map = F.relu(self.features(x), inplace=False)
        lung_mask = lung_mask.to(device=feature_map.device, dtype=feature_map.dtype)
        disease_prior_maps = disease_prior_maps.to(
            device=feature_map.device, dtype=feature_map.dtype
        )
        if lung_mask.shape[-2:] != feature_map.shape[-2:]:
            lung_mask = F.interpolate(
                lung_mask, size=feature_map.shape[-2:], mode="nearest"
            )
        if disease_prior_maps.shape[-2:] != feature_map.shape[-2:]:
            disease_prior_maps = F.interpolate(
                disease_prior_maps, size=feature_map.shape[-2:], mode="bilinear",
                align_corners=False,
            )

        logits = []
        for index in range(self.num_classes):
            lung_features = self.lung_apams[index](feature_map, lung_mask)
            prior_features = self.prior_apams[index](
                feature_map, disease_prior_maps[:, index:index + 1]
            )
            combined_features = torch.cat((lung_features, prior_features), dim=1)
            pooled_features = F.adaptive_avg_pool2d(combined_features, 1).flatten(1)
            logits.append(self.classifiers[index](pooled_features))

        return torch.cat(logits, dim=1)
