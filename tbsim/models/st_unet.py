# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Tuple, Optional
#
#
# # =========================================================
# # Basic blocks
# # =========================================================
#
# class ConvBNReLU(nn.Module):
#     def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
#         super().__init__()
#         if p is None:
#             p = k // 2
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.block(x)
#
#
# class ResBlock2D(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=1):
#         super().__init__()
#         self.conv1 = ConvBNReLU(in_ch, out_ch, k=3, s=stride)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#         )
#
#         if in_ch != out_ch or stride != 1:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_ch),
#             )
#         else:
#             self.shortcut = nn.Identity()
#
#         self.act = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         identity = self.shortcut(x)
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = out + identity
#         return self.act(out)
#
#
# class EncoderStage2D(nn.Module):
#     def __init__(self, in_ch, out_ch, downsample):
#         super().__init__()
#         stride = 2 if downsample else 1
#         self.block1 = ResBlock2D(in_ch, out_ch, stride=stride)
#         self.block2 = ResBlock2D(out_ch, out_ch, stride=1)
#
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         return x
#
#
# # =========================================================
# # Shared frame-wise encoder
# # Input: (B, T, 1, H, W)
# # =========================================================
#
# class SharedFrameEncoder2D(nn.Module):
#     def __init__(self, in_channels=1, channels=(64, 128, 256, 512)):
#         super().__init__()
#         c1, c2, c3, c4 = channels
#
#         self.stem = nn.Sequential(
#             ConvBNReLU(in_channels, c1, k=7, s=2, p=3),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )
#
#         self.layer1 = EncoderStage2D(c1, c1, downsample=False)
#         self.layer2 = EncoderStage2D(c1, c2, downsample=True)
#         self.layer3 = EncoderStage2D(c2, c3, downsample=True)
#         self.layer4 = EncoderStage2D(c3, c4, downsample=True)
#
#         self.out_channels = {
#             "layer1": c1,
#             "layer2": c2,
#             "layer3": c3,
#             "layer4": c4,
#         }
#
#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         # x: (B, T, C, H, W)
#         b, t, c, h, w = x.shape
#         x = x.reshape(b * t, c, h, w)
#
#         x = self.stem(x)
#         l1 = self.layer1(x)
#         l2 = self.layer2(l1)
#         l3 = self.layer3(l2)
#         l4 = self.layer4(l3)
#
#         def restore(feat):
#             _, ch, hh, ww = feat.shape
#             return feat.view(b, t, ch, hh, ww)
#
#         return {
#             "layer1": restore(l1),
#             "layer2": restore(l2),
#             "layer3": restore(l3),
#             "layer4": restore(l4),
#         }
#
#
# # =========================================================
# # Static map encoder (2D only, processed once)
# # =========================================================
#
# class StaticMapEncoder2D(nn.Module):
#     def __init__(self, in_channels=3, channels=(64, 128, 256, 512)):
#         super().__init__()
#         c1, c2, c3, c4 = channels
#
#         self.stem = nn.Sequential(
#             ConvBNReLU(in_channels, c1, k=7, s=2, p=3),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )
#
#         self.layer1 = EncoderStage2D(c1, c1, downsample=False)
#         self.layer2 = EncoderStage2D(c1, c2, downsample=True)
#         self.layer3 = EncoderStage2D(c2, c3, downsample=True)
#         self.layer4 = EncoderStage2D(c3, c4, downsample=True)
#
#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         x = self.stem(x)
#         l1 = self.layer1(x)
#         l2 = self.layer2(l1)
#         l3 = self.layer3(l2)
#         l4 = self.layer4(l3)
#         return {
#             "layer1": l1,
#             "layer2": l2,
#             "layer3": l3,
#             "layer4": l4,
#         }
#
#
# # =========================================================
# # Lightweight factorized spatiotemporal bottleneck
# # Input: (B, C, T, H, W)
# # =========================================================
#
# class FactorizedSTBlock(nn.Module):
#     def __init__(self, channels, temporal_kernel=3, expansion=2, dropout=0.0):
#         super().__init__()
#         hidden = channels * expansion
#
#         self.spatial = nn.Sequential(
#             nn.Conv3d(channels, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
#             nn.BatchNorm3d(hidden),
#             nn.ReLU(inplace=True),
#         )
#
#         self.temporal = nn.Sequential(
#             nn.Conv3d(hidden, hidden, kernel_size=(temporal_kernel, 1, 1),
#                       padding=(temporal_kernel // 2, 0, 0), bias=False),
#             nn.BatchNorm3d(hidden),
#             nn.ReLU(inplace=True),
#         )
#
#         self.proj = nn.Sequential(
#             nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
#             nn.BatchNorm3d(channels),
#         )
#
#         self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
#         self.act = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         identity = x
#         out = self.spatial(x)
#         out = self.temporal(out)
#         out = self.proj(out)
#         out = self.dropout(out)
#         out = out + identity
#         return self.act(out)
#
#
# class STBottleneck(nn.Module):
#     def __init__(self, channels, num_blocks=3, temporal_kernel=3):
#         super().__init__()
#         self.blocks = nn.Sequential(*[
#             FactorizedSTBlock(channels, temporal_kernel=temporal_kernel)
#             for _ in range(num_blocks)
#         ])
#
#     def forward(self, x):
#         return self.blocks(x)
#
#
# # class TemporalAggregator(nn.Module):
# #     """
# #     Advanced Temporal Aggregator for Skip Connections
# #     Input: (B, T, C, H, W)
# #     Output: (B, C, H, W)
# #     """
# #
# #     def __init__(self, channels, seq_len=None, mode="attention"):
# #         super().__init__()
# #         self.mode = mode
# #
# #         if mode == "attention":
# #             # Temporal Attention Pooling (TAP)
# #             # Learns a specific weight for each time step per pixel
# #             self.attention = nn.Sequential(
# #                 nn.Conv3d(channels, 1, kernel_size=1, bias=False),
# #                 nn.Softmax(dim=2)
# #             )
# #             # Post-aggregation projection
# #             self.proj = nn.Sequential(
# #                 nn.Conv2d(channels, channels, kernel_size=1, bias=False),
# #                 nn.BatchNorm2d(channels),
# #                 nn.ReLU(inplace=True)
# #             )
# #
# #         elif mode == "conv3d":
# #             if seq_len is None:
# #                 raise ValueError("seq_len must be provided for conv3d mode.")
# #             # 1x1xT Temporal Convolution
# #             # Reduces the T dimension to 1 using learned chronological weights
# #             self.temporal_conv = nn.Sequential(
# #                 nn.Conv3d(channels, channels, kernel_size=(seq_len, 1, 1), bias=False),
# #                 nn.BatchNorm3d(channels),
# #                 nn.ReLU(inplace=True)
# #             )
# #         else:
# #             raise ValueError(f"Unsupported mode: {mode}. Use 'attention' or 'conv3d'.")
# #
# #     def forward(self, x):
# #         # Input shape: (B, T, C, H, W)
# #         x = x.permute(0, 2, 1, 3, 4).contiguous()  # Reshape to: (B, C, T, H, W)
# #
# #         if self.mode == "attention":
# #             # 1. Calculate attention weights over the T dimension
# #             attn_weights = self.attention(x)  # (B, 1, T, H, W)
# #
# #             # 2. Element-wise multiplication and sum over T (Weighted Average)
# #             x = torch.sum(x * attn_weights, dim=2)  # (B, C, H, W)
# #
# #             # 3. Final feature projection
# #             x = self.proj(x)
# #
# #         elif self.mode == "conv3d":
# #             # 1. Convolve across the entire T dimension to reduce it to 1
# #             x = self.temporal_conv(x)  # (B, C, 1, H, W)
# #
# #             # 2. Drop the temporal dimension
# #             x = x.squeeze(2)  # (B, C, H, W)
# #
# #         return x
#
#
# # =========================================================
# # Temporal aggregation for skip connections
# # Input: (B, T, C, H, W)
# # Output: (B, C, H, W)
# # =========================================================
#
# class TemporalAggregator(nn.Module):
#     def __init__(self, channels, mode="conv"):
#         super().__init__()
#         self.mode = mode
#
#         if mode == "conv":
#             self.temporal_mix = nn.Sequential(
#                 nn.Conv3d(channels, channels, kernel_size=(3, 1, 1),
#                           padding=(1, 0, 0), groups=channels, bias=False),
#                 nn.BatchNorm3d(channels),
#                 nn.ReLU(inplace=True),
#                 nn.Conv3d(channels, channels, kernel_size=1, bias=False),
#                 nn.BatchNorm3d(channels),
#                 nn.ReLU(inplace=True),
#             )
#         elif mode not in {"mean", "max"}:
#             raise ValueError(f"Unsupported mode: {mode}")
#
#     def forward(self, x):
#         # x: (B, T, C, H, W)
#         x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
#
#         if self.mode == "conv":
#             x = self.temporal_mix(x)
#             x = x.mean(dim=2)
#         elif self.mode == "mean":
#             x = x.mean(dim=2)
#         else:
#             x = x.max(dim=2).values
#
#         return x
#
#
# # =========================================================
# # Decoder
# # =========================================================
#
# class UpBlock(nn.Module):
#     def __init__(self, in_ch, skip_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             ConvBNReLU(in_ch + skip_ch, out_ch, k=3, s=1),
#             ConvBNReLU(out_ch, out_ch, k=3, s=1),
#         )
#
#     def forward(self, x, skip):
#         x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
#         x = torch.cat([x, skip], dim=1)
#         return self.conv(x)
#
#
# class STUNetDecoder(nn.Module):
#     def __init__(self, bottleneck_ch, skip_channels, out_channels):
#         super().__init__()
#         c1, c2, c3 = skip_channels
#
#         self.pre = ConvBNReLU(bottleneck_ch, 512, k=3, s=1)
#         self.up1 = UpBlock(512, c3, 256)
#         self.up2 = UpBlock(256, c2, 128)
#         self.up3 = UpBlock(128, c1, 64)
#         self.head = nn.Sequential(
#             ConvBNReLU(64, 64, k=3, s=1),
#             nn.Conv2d(64, out_channels, kernel_size=1),
#         )
#
#     def forward(self, bottleneck_2d, skip1, skip2, skip3):
#         x = self.pre(bottleneck_2d)
#         x = self.up1(x, skip3)
#         x = self.up2(x, skip2)
#         x = self.up3(x, skip1)
#         return self.head(x)
#
#
# # =========================================================
# # Main ST-UNet encoder
# # =========================================================
#
# class STUNetMapEncoder(nn.Module):
#     """
#     Tailored for:
#       dynamic_inputs: (B, 31, 1, H, W)
#       static_map:     (B, C_map, H, W)
#
#     Output:
#       fc_out: optional global feature
#       feat_map_out: dense feature map for diffusion conditioning
#     """
#     def __init__(
#         self,
#         static_map_channels=3,
#         encoder_channels=(64, 128, 256, 512),
#         bottleneck_blocks=3,
#         bottleneck_temporal_kernel=3,
#         grid_feature_dim=128,
#         global_feature_dim=None,
#         use_static_branch=True,
#         skip_agg_mode="conv",
#         fuse_static_add=True,
#     ):
#         super().__init__()
#
#         self.return_global_feat = global_feature_dim is not None
#         self.return_grid_feat = grid_feature_dim is not None
#         self.use_static_branch = use_static_branch
#         self.fuse_static_add = fuse_static_add
#
#         # dynamic occupancy raster: single channel per timestep
#         self.dynamic_encoder = SharedFrameEncoder2D(
#             in_channels=1,
#             channels=encoder_channels,
#         )
#
#         if use_static_branch:
#             self.static_encoder = StaticMapEncoder2D(
#                 in_channels=static_map_channels,
#                 channels=encoder_channels,
#             )
#
#         c1, c2, c3, c4 = encoder_channels
#
#         self.bottleneck = STBottleneck(
#             channels=c4,
#             num_blocks=bottleneck_blocks,
#             temporal_kernel=bottleneck_temporal_kernel,
#         )
#
#         self.skip1_agg = TemporalAggregator(c1, mode=skip_agg_mode)
#         self.skip2_agg = TemporalAggregator(c2, mode=skip_agg_mode)
#         self.skip3_agg = TemporalAggregator(c3, mode=skip_agg_mode)
#         self.bottleneck_agg = TemporalAggregator(c4, mode=skip_agg_mode)
#
#         # self.skip1_agg = TemporalAggregator(c1, seq_len=31)
#         # self.skip2_agg = TemporalAggregator(c2, seq_len=31)
#         # self.skip3_agg = TemporalAggregator(c3, seq_len=31)
#         # self.bottleneck_agg = TemporalAggregator(c4, seq_len=31)
#
#         if self.return_grid_feat:
#             self.decoder = STUNetDecoder(
#                 bottleneck_ch=c4,
#                 skip_channels=(c1, c2, c3),
#                 out_channels=grid_feature_dim,
#             )
#
#         if self.return_global_feat:
#             self.global_head = nn.Sequential(
#                 nn.AdaptiveAvgPool3d((1, 1, 1)),
#                 nn.Flatten(),
#                 nn.Linear(c4, global_feature_dim),
#             )
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#
#     def _fuse_static(self, dyn_feats, static_feats):
#         fused = {}
#         for k in dyn_feats.keys():
#             d = dyn_feats[k]                        # (B, T, C, H, W)
#             s = static_feats[k].unsqueeze(1)       # (B, 1, C, H, W)
#
#             if self.fuse_static_add:
#                 fused[k] = d + s
#             else:
#                 # Simple fallback: still use add unless you later want concat+proj
#                 fused[k] = d + s
#         return fused
#
#     def forward(
#         self,
#         dynamic_inputs: torch.Tensor,
#         static_map: Optional[torch.Tensor] = None,
#         align_fn=None,
#     ):
#         """
#         dynamic_inputs: (B, 31, 1, H, W)
#         static_map:     (B, C_map, H, W)
#         """
#         if align_fn is not None:
#             dynamic_inputs = align_fn(dynamic_inputs)
#
#         dyn_feats = self.dynamic_encoder(dynamic_inputs)
#
#         if self.use_static_branch and static_map is not None:
#             static_feats = self.static_encoder(static_map)
#             dyn_feats = self._fuse_static(dyn_feats, static_feats)
#
#         l1 = dyn_feats["layer1"]                                # (B, T, C1, H1, W1)
#         l2 = dyn_feats["layer2"]
#         l3 = dyn_feats["layer3"]
#         l4 = dyn_feats["layer4"].permute(0, 2, 1, 3, 4)        # (B, C4, T, H4, W4)
#
#         l4 = self.bottleneck(l4)
#
#         fc_out = self.global_head(l4) if self.return_global_feat else None
#
#         feat_map_out = None
#         if self.return_grid_feat:
#             bottleneck_2d = self.bottleneck_agg(
#                 l4.permute(0, 2, 1, 3, 4).contiguous()
#             )                                                  # (B, C4, H4, W4)
#
#             skip1 = self.skip1_agg(l1)                         # (B, C1, H1, W1)
#             skip2 = self.skip2_agg(l2)
#             skip3 = self.skip3_agg(l3)
#
#             feat_map_out = self.decoder(
#                 bottleneck_2d=bottleneck_2d,
#                 skip1=skip1,
#                 skip2=skip2,
#                 skip3=skip3,
#             )
#
#         feat_map_out = self.pool(feat_map_out).squeeze(-1).squeeze(-1)
#         return fc_out, feat_map_out
#
#
# # =========================================================
# # Example
# # =========================================================
#
# if __name__ == "__main__":
#     B, T, H, W = 2, 31, 224, 224
#     C_map = 3
#
#     model = STUNetMapEncoder(
#         static_map_channels=C_map,
#         encoder_channels=(64, 128, 256, 512),
#         bottleneck_blocks=3,
#         bottleneck_temporal_kernel=3,
#         grid_feature_dim=128,
#         global_feature_dim=256,
#         use_static_branch=True,
#         skip_agg_mode="conv",
#     )
#
#     # single-channel occupancy raster sequence
#     dynamic_inputs = torch.randn(B, T, 1, H, W)
#
#     # optional static map channels
#     static_map = torch.randn(B, C_map, H, W)
#
#     fc_out, feat_map_out = model(dynamic_inputs, static_map)
#
#     print("fc_out:", None if fc_out is None else fc_out.shape)
#     print("feat_map_out:", None if feat_map_out is None else feat_map_out.shape)



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


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
# Decoder
# =========================================================

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch, k=3, s=1),
            ConvBNReLU(out_ch, out_ch, k=3, s=1),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class STUNetDecoder(nn.Module):
    def __init__(self, bottleneck_ch, skip_channels, out_channels):
        super().__init__()
        c1, c2, c3 = skip_channels

        self.pre = ConvBNReLU(bottleneck_ch, 512, k=3, s=1)
        self.up1 = UpBlock(512, c3, 256)
        self.up2 = UpBlock(256, c2, 128)
        self.up3 = UpBlock(128, c1, 64)
        self.head = nn.Sequential(
            ConvBNReLU(64, 64, k=3, s=1),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, bottleneck_2d, skip1, skip2, skip3):
        x = self.pre(bottleneck_2d)
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        return self.head(x)


# =========================================================
# Main ST-UNet encoder
# =========================================================

class STUNetMapEncoder(nn.Module):
    """
    Input:
        image_sequence: (B, T, C_img, H, W)

    Output:
        fc_out: optional global feature
        feat_map_out: dense feature map (B, C_out, H_out, W_out)
    """
    def __init__(
        self,
        image_channels=3,
        encoder_channels=(64, 128, 256, 512),
        bottleneck_blocks=3,
        bottleneck_temporal_kernel=3,
        grid_feature_dim=128,
        global_feature_dim=None,
        skip_agg_mode="conv",
    ):
        super().__init__()

        self.return_global_feat = global_feature_dim is not None
        self.return_grid_feat = grid_feature_dim is not None

        self.encoder = SharedFrameEncoder2D(
            in_channels=image_channels,
            channels=encoder_channels,
        )

        c1, c2, c3, c4 = encoder_channels

        self.bottleneck = STBottleneck(
            channels=c4,
            num_blocks=bottleneck_blocks,
            temporal_kernel=bottleneck_temporal_kernel,
        )

        self.skip1_agg = TemporalAggregator(c1, mode=skip_agg_mode)
        self.skip2_agg = TemporalAggregator(c2, mode=skip_agg_mode)
        self.skip3_agg = TemporalAggregator(c3, mode=skip_agg_mode)
        self.bottleneck_agg = TemporalAggregator(c4, mode=skip_agg_mode)

        if self.return_grid_feat:
            self.decoder = STUNetDecoder(
                bottleneck_ch=c4,
                skip_channels=(c1, c2, c3),
                out_channels=grid_feature_dim,
            )

        if self.return_global_feat:
            self.global_head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(c4, global_feature_dim),
            )

    def forward(
        self,
        image_sequence: torch.Tensor,
        align_fn=None,
    ):
        """
        image_sequence: (B, T, C_img, H, W)
        """
        if align_fn is not None:
            image_sequence = align_fn(image_sequence)

        feats = self.encoder(image_sequence)

        l1 = feats["layer1"]                                  # (B, T, C1, H1, W1)
        l2 = feats["layer2"]
        l3 = feats["layer3"]
        l4 = feats["layer4"].permute(0, 2, 1, 3, 4).contiguous()   # (B, C4, T, H4, W4)

        l4 = self.bottleneck(l4)

        fc_out = self.global_head(l4) if self.return_global_feat else None

        feat_map_out = None
        if self.return_grid_feat:
            bottleneck_2d = self.bottleneck_agg(
                l4.permute(0, 2, 1, 3, 4).contiguous()
            )

            skip1 = self.skip1_agg(l1)
            skip2 = self.skip2_agg(l2)
            skip3 = self.skip3_agg(l3)

            feat_map_out = self.decoder(
                bottleneck_2d=bottleneck_2d,
                skip1=skip1,
                skip2=skip2,
                skip3=skip3,
            )

        return fc_out, feat_map_out


# =========================================================
# Example
# =========================================================

if __name__ == "__main__":
    B, T, H, W = 2, 31, 224, 224
    C_img = 3  # example: RGB-like superimposed frame

    model = STUNetMapEncoder(
        image_channels=C_img,
        encoder_channels=(64, 128, 256, 512),
        bottleneck_blocks=3,
        bottleneck_temporal_kernel=3,
        grid_feature_dim=128,
        global_feature_dim=256,
    )

    image_sequence = torch.randn(B, T, C_img, H, W)

    fc_out, feat_map_out = model(image_sequence)

    print("fc_out:", None if fc_out is None else fc_out.shape)
    print("feat_map_out:", None if feat_map_out is None else feat_map_out.shape)