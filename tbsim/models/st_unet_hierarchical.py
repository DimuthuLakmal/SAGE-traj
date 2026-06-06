import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# =========================================================
# Reuse your existing blocks:
# =========================================================
# Basic blocks
# =========================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, k=3, s=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.act(out)


class EncoderStage2D(nn.Module):
    def __init__(self, in_ch, out_ch, downsample):
        super().__init__()
        stride = 2 if downsample else 1
        self.block1 = ResBlock2D(in_ch, out_ch, stride=stride)
        self.block2 = ResBlock2D(out_ch, out_ch, stride=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


# =========================================================
# Shared frame-wise encoder
# Input: (B, T, C, H, W)
# =========================================================

class SharedFrameEncoder2D(nn.Module):
    def __init__(self, in_channels=3, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, c1, k=7, s=2, p=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = EncoderStage2D(c1, c1, downsample=False)
        self.layer2 = EncoderStage2D(c1, c2, downsample=True)
        self.layer3 = EncoderStage2D(c2, c3, downsample=True)
        self.layer4 = EncoderStage2D(c3, c4, downsample=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)

        x = self.stem(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        def restore(feat):
            _, ch, hh, ww = feat.shape
            return feat.view(b, t, ch, hh, ww)

        return {
            "layer1": restore(l1),
            "layer2": restore(l2),
            "layer3": restore(l3),
            "layer4": restore(l4),
        }


# =========================================================
# Lightweight factorized spatiotemporal bottleneck
# Input: (B, C, T, H, W)
# =========================================================

class FactorizedSTBlock(nn.Module):
    def __init__(self, channels, temporal_kernel=3, expansion=2, dropout=0.0):
        super().__init__()
        hidden = channels * expansion

        self.spatial = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
        )

        self.temporal = nn.Sequential(
            nn.Conv3d(
                hidden, hidden,
                kernel_size=(temporal_kernel, 1, 1),
                padding=(temporal_kernel // 2, 0, 0),
                bias=False
            ),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Sequential(
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        )

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.spatial(x)
        out = self.temporal(out)
        out = self.proj(out)
        out = self.dropout(out)
        out = out + identity
        return self.act(out)


class STBottleneck(nn.Module):
    def __init__(self, channels, num_blocks=3, temporal_kernel=3):
        super().__init__()
        self.blocks = nn.Sequential(*[
            FactorizedSTBlock(channels, temporal_kernel=temporal_kernel)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)


# =========================================================
# Temporal aggregation for skip connections
# Input: (B, T, C, H, W)
# Output: (B, C, H, W)
# =========================================================

class TemporalAggregator(nn.Module):
    def __init__(self, channels, mode="conv"):
        super().__init__()
        self.mode = mode

        if mode == "conv":
            self.temporal_mix = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=(3, 1, 1),
                          padding=(1, 0, 0), groups=channels, bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
            )
        elif mode not in {"mean", "max"}:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x):
        # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)

        if self.mode == "conv":
            x = self.temporal_mix(x)
            x = x.mean(dim=2)
        elif self.mode == "mean":
            x = x.mean(dim=2)
        else:
            x = x.max(dim=2).values

        return x

# =========================================================


class LightTemporalBlock(nn.Module):
    """
    Lightweight temporal refinement for high-resolution stages.
    Input:  (B, T, C, H, W)
    Output: (B, T, C, H, W)
    """
    def __init__(self, channels, temporal_kernel=3, expansion=1):
        super().__init__()
        hidden = channels * expansion

        self.block = nn.Sequential(
            nn.Conv3d(
                channels, channels,
                kernel_size=(temporal_kernel, 1, 1),
                padding=(temporal_kernel // 2, 0, 0),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, T, C, H, W)
        identity = x
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # (B, C, T, H, W)
        out = self.block(x)
        out = out + x
        out = self.act(out)
        out = out.permute(0, 2, 1, 3, 4).contiguous()  # back to (B, T, C, H, W)
        return out


class MultiScaleVectorHead(nn.Module):
    def __init__(self, in_dims, out_dim):
        super().__init__()
        total_dim = sum(in_dims)

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.ReLU(inplace=True),
            nn.Linear(total_dim, out_dim),
        )

    def forward(self, feats):
        # feats: list of (B, C, H, W)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        x = torch.cat(pooled, dim=1)
        return self.mlp(x)


class STUNetTemporalPyramidEncoder(nn.Module):
    """
    Multi-scale temporal fusion encoder for diffusion conditioning.

    Outputs:
        global_out: optional global feature vector
        fused_map:  fused 2D conditioning map
    """
    def __init__(
        self,
        image_channels=3,
        encoder_channels=(64, 128, 256, 512),
        bottleneck_blocks=3,
        bottleneck_temporal_kernel=3,
        fusion_dim=128,
        global_feature_dim=None,
        agg_mode="conv",
        use_l1=True,
        use_l2=True,
        use_l3=True,
        use_l4=True,
    ):
        super().__init__()

        c1, c2, c3, c4 = encoder_channels
        self.global_feature_dim = global_feature_dim

        self.use_l1 = use_l1
        self.use_l2 = use_l2
        self.use_l3 = use_l3
        self.use_l4 = use_l4

        self.encoder = SharedFrameEncoder2D(
            in_channels=image_channels,
            channels=encoder_channels,
        )

        # Temporal refinement at upper scales
        if use_l1:
            self.temporal_l1 = LightTemporalBlock(c1, temporal_kernel=3)
            self.agg_l1 = TemporalAggregator(c1, mode=agg_mode)

        if use_l2:
            self.temporal_l2 = LightTemporalBlock(c2, temporal_kernel=3)
            self.agg_l2 = TemporalAggregator(c2, mode=agg_mode)

        if use_l3:
            self.temporal_l3 = LightTemporalBlock(c3, temporal_kernel=3)
            self.agg_l3 = TemporalAggregator(c3, mode=agg_mode)

        # Strong temporal modeling at bottleneck
        if use_l4:
            self.bottleneck = STBottleneck(
                channels=c4,
                num_blocks=bottleneck_blocks,
                temporal_kernel=bottleneck_temporal_kernel,
            )
            self.agg_l4 = TemporalAggregator(c4, mode=agg_mode)

        fusion_dims = []
        if use_l1: fusion_dims.append(c1)
        if use_l2: fusion_dims.append(c2)
        if use_l3: fusion_dims.append(c3)
        if use_l4: fusion_dims.append(c4)

        self.vector_head = MultiScaleVectorHead(
            in_dims=fusion_dims,
            out_dim=global_feature_dim
        )

        if global_feature_dim is not None:
            self.global_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(fusion_dim, global_feature_dim),
            )

    def forward(self, image_sequence: torch.Tensor, align_fn=None):
        if align_fn is not None:
            image_sequence = align_fn(image_sequence)

        feats = self.encoder(image_sequence)
        l1 = feats["layer1"]   # (B, T, C1, H1, W1)
        l2 = feats["layer2"]
        l3 = feats["layer3"]
        l4 = feats["layer4"]

        multi_scale_vec_inputs = []

        if self.use_l1:
            l1t = self.temporal_l1(l1)
            f1 = self.agg_l1(l1t)
            multi_scale_vec_inputs.append(f1)

        if self.use_l2:
            l2t = self.temporal_l2(l2)
            f2 = self.agg_l2(l2t)
            multi_scale_vec_inputs.append(f2)

        if self.use_l3:
            l3t = self.temporal_l3(l3)
            f3 = self.agg_l3(l3t)
            multi_scale_vec_inputs.append(f3)

        if self.use_l4:
            l4t = l4.permute(0, 2, 1, 3, 4)
            l4t = self.bottleneck(l4t)
            l4t = l4t.permute(0, 2, 1, 3, 4)
            f4 = self.agg_l4(l4t)
            multi_scale_vec_inputs.append(f4)

        global_out = self.vector_head(multi_scale_vec_inputs)

        return global_out


if __name__ == "__main__":
    import torch

    # =========================
    # Config
    # =========================
    B, T, H, W = 2, 31, 224, 224
    C_img = 3

    # =========================
    # Model
    # =========================
    model = STUNetTemporalPyramidEncoder(
        image_channels=C_img,
        encoder_channels=(64, 128, 256, 512),
        bottleneck_blocks=3,
        bottleneck_temporal_kernel=3,
        global_feature_dim=256,
        use_l1=True,
        use_l2=True,
        use_l3=True,
        use_l4=True,
    )

    # =========================
    # Dummy input
    # =========================
    image_sequence = torch.randn(B, T, C_img, H, W)

    # =========================
    # Forward pass
    # =========================
    global_out = model(image_sequence)

    # =========================
    # Debug prints
    # =========================
    print("Input shape:", image_sequence.shape)

    print("\n--- Outputs ---")
    print("global_out shape:", None if global_out is None else global_out.shape)

    # =========================
    # Sanity checks
    # =========================
    assert global_out is not None, "Global output should not be None"
    assert global_out.shape == (B, 256), "Unexpected global feature shape"

    print("\nSanity check passed ✅")