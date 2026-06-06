import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from tbsim.models.st_unet_hierarchical import STUNetTemporalPyramidEncoder

class LightweightGoalCVAE(nn.Module):
    def __init__(self, input_image_shape, feat_dim=128, latent_dim=4, hidden_dim=128):
        super().__init__()

        self.image_encoder = STUNetTemporalPyramidEncoder(image_channels=3,
                encoder_channels=(64, 128, 256, 512),
                bottleneck_blocks=3,
                bottleneck_temporal_kernel=3,
                global_feature_dim=128,
                use_l1=True,
                use_l2=True,
                use_l3=True,
                use_l4=True)

        # p(z | image)
        self.prior_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # q(z | image, goal)
        # goal = [x, y, yaw]
        self.posterior_net = nn.Sequential(
            nn.Linear(feat_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # p(goal | image, z)
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim + latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # x, y, yaw
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def split_mu_logvar(params):
        mu, logvar = torch.chunk(params, chunks=2, dim=-1)
        logvar = torch.clamp(logvar, min=-8.0, max=4.0)
        return mu, logvar

    def forward(self, image, goal=None, num_samples=None):
        """
        image: [B, C, H, W]
        goal: optional dict with goal_position [B,2], goal_yaw [B,1]
        num_samples: if None, returns one sample. Otherwise returns N samples.
        """
        b = image.shape[0]
        feat = self.image_encoder(image)

        prior_mu, prior_logvar = self.split_mu_logvar(self.prior_net(feat))

        if goal is not None:
            goal_vec = torch.cat(
                [goal["goal_position"], goal["goal_yaw"]],
                dim=-1
            )  # [B, 3]

            post_mu, post_logvar = self.split_mu_logvar(
                self.posterior_net(torch.cat([feat, goal_vec], dim=-1))
            )

            z = self.reparameterize(post_mu, post_logvar)
            pred = self.decoder(torch.cat([feat, z], dim=-1))  # [B, 3]

            return {
                "pred_goal": pred,
                "prior_mu": prior_mu,
                "prior_logvar": prior_logvar,
                "post_mu": post_mu,
                "post_logvar": post_logvar,
            }

        # inference: sample from prior
        n = num_samples or 1
        feat_rep = feat[:, None].expand(b, n, feat.shape[-1]).reshape(b * n, -1)
        prior_mu_rep = prior_mu[:, None].expand(b, n, prior_mu.shape[-1]).reshape(b * n, -1)
        prior_logvar_rep = prior_logvar[:, None].expand(b, n, prior_logvar.shape[-1]).reshape(b * n, -1)

        z = self.reparameterize(prior_mu_rep, prior_logvar_rep)
        pred = self.decoder(torch.cat([feat_rep, z], dim=-1))
        pred = pred.reshape(b, n, 3)

        return {
            "pred_goal": pred,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
        }