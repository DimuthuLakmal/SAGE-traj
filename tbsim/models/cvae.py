import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import numpy as np

from tbsim.utils.internvl_utils import get_item, generate_llm_output


class LightweightTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        model_dim=256,
        feat_dim=256,
        num_layers=2,
        num_heads=4,
        ff_dim=512,
        max_seq_len=128,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len + 1, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [B, T, 2048]
        attention_mask: [B, T], 1 = valid, 0 = padding
        """
        b, t, _ = hidden_states.shape

        x = self.input_proj(hidden_states)  # [B, T, model_dim]

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)      # [B, T+1, model_dim]

        x = x + self.pos_emb[:, :t + 1]

        if attention_mask is not None:
            cls_mask = torch.ones(
                b, 1,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

            # PyTorch Transformer expects True for padded positions
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        x = self.transformer(
            x,
            src_key_padding_mask=key_padding_mask,
        )

        cls_feat = x[:, 0]  # [B, model_dim]

        feat = self.output_proj(cls_feat)  # [B, feat_dim]

        return feat


class SoftSectionAttentionEncoder(nn.Module):
    """
    Soft section-aware encoder for MLLM hidden states.

    Instead of requiring explicit text boundaries, this module uses multiple
    learnable semantic queries to attend over the full hidden-state sequence.

    Input:
        hidden_states: [B, T, D]
        attention_mask: optional [B, T]

    Output:
        final_feat: [B, feat_dim]
        section_feats: [B, num_sections, section_dim]
        token_attn: [B, num_sections, T]
        section_weights: [B, num_sections]
    """
    def __init__(
        self,
        input_dim,
        feat_dim=128,
        attn_dim=128,
        num_sections=4,
        dropout=0.1,
    ):
        super().__init__()

        self.num_sections = num_sections

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Four semantic queries.
        # They are not hard-coded to text sections, but can learn different roles.
        self.section_queries = nn.Parameter(
            torch.randn(1, num_sections, attn_dim)
        )

        self.key_proj = nn.Linear(attn_dim, attn_dim)
        self.value_proj = nn.Linear(attn_dim, attn_dim)

        # Optional section identity embedding.
        self.section_type_embedding = nn.Parameter(
            torch.randn(1, num_sections, attn_dim)
        )

        # Fuse section-level features adaptively.
        self.section_gate = nn.Sequential(
            nn.Linear(attn_dim * num_sections, num_sections),
            nn.Softmax(dim=-1)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(attn_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [B, T, D]
        attention_mask: optional [B, T], 1 = valid token, 0 = padding
        """
        b, t, _ = hidden_states.shape

        x = self.input_proj(hidden_states)  # [B, T, attn_dim]

        q = self.section_queries.expand(b, -1, -1)  # [B, 4, attn_dim]
        k = self.key_proj(x)                        # [B, T, attn_dim]
        v = self.value_proj(x)                      # [B, T, attn_dim]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        # [B, 4, T]

        if attention_mask is not None:
            mask = attention_mask[:, None, :].bool()  # [B, 1, T]
            attn_scores = attn_scores.masked_fill(~mask, -1e9)

        token_attn = torch.softmax(attn_scores, dim=-1)  # [B, 4, T]

        # Four soft semantic features.
        section_feats = torch.matmul(token_attn, v)  # [B, 4, attn_dim]

        # Add role identity.
        section_feats = section_feats + self.section_type_embedding

        # Adaptive section fusion.
        flat = section_feats.flatten(start_dim=1)  # [B, 4 * attn_dim]
        section_weights = self.section_gate(flat)  # [B, 4]

        fused = torch.sum(
            section_feats * section_weights.unsqueeze(-1),
            dim=1
        )  # [B, attn_dim]

        final_feat = self.output_proj(fused)  # [B, feat_dim]

        return final_feat


class AttentionPoolingEncoder(nn.Module):
    """
    Lightweight sequence encoder for MLLM hidden states.
    Input:  hidden_states [B, T, D]
    Output: scene feature [B, feat_dim]
    """
    def __init__(self, input_dim, feat_dim=128, attn_dim=128, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Learnable query used to attend over the hidden-state sequence
        self.query = nn.Parameter(torch.randn(1, 1, attn_dim))

        self.key_proj = nn.Linear(attn_dim, attn_dim)
        self.value_proj = nn.Linear(attn_dim, attn_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(attn_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [B, T, D]
        attention_mask: optional [B, T], where 1 = valid token, 0 = padding
        """
        b, t, _ = hidden_states.shape

        x = self.input_proj(hidden_states)  # [B, T, attn_dim]

        q = self.query.expand(b, -1, -1)    # [B, 1, attn_dim]
        k = self.key_proj(x)                # [B, T, attn_dim]
        v = self.value_proj(x)              # [B, T, attn_dim]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        # [B, 1, T]

        if attention_mask is not None:
            mask = attention_mask[:, None, :].bool()  # [B, 1, T]
            attn_scores = attn_scores.masked_fill(~mask, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, 1, T]

        pooled = torch.matmul(attn_weights, v).squeeze(1)   # [B, attn_dim]

        feat = self.output_proj(pooled)  # [B, feat_dim]

        return feat


class LightweightGoalCVAE(nn.Module):
    def __init__(
        self,
        mllm_hidden_dim,
        feat_dim=128,
        latent_dim=4,
        hidden_dim=128,
        attn_dim=128,
        dropout=0.1,
        llm_model=None, 
        llm_tokenizer=None,
    ):
        super().__init__()

        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.llm_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0,
            return_dict_in_generate=True,
            output_logits=False,
            output_scores=False,
            output_hidden_states=True
        )

        self.sequence_encoder = SoftSectionAttentionEncoder(
            input_dim=mllm_hidden_dim,
            feat_dim=feat_dim,
            attn_dim=attn_dim,
            dropout=dropout,
        )

        # self.sequence_encoder = LightweightTransformerEncoder(
        #     input_dim=mllm_hidden_dim,
        #     model_dim=256,
        #     feat_dim=256,
        #     num_layers=2,
        #     num_heads=4,
        #     ff_dim=512,
        #     max_seq_len=512,
        #     dropout=0.1,
        # )

        # p(z | h)
        self.prior_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # q(z | h, goal)
        # goal = [x, y, yaw]
        self.posterior_net = nn.Sequential(
            nn.Linear(feat_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # p(goal | h, z)
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim + latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def _bundle_tensors(self, tensor_list, max_seq_len=1024, pad_value=0.0):
        batch_size = len(tensor_list)
        feature_dim = tensor_list[0].shape[1]

        # Create output tensor
        batch_tensor = torch.full(
            (batch_size, max_seq_len, feature_dim),
            fill_value=pad_value,
            dtype=tensor_list[0].dtype,
            device=tensor_list[0].device,
        )

        mask = torch.zeros(
                (batch_size, max_seq_len),
                dtype=torch.bool,
                device=tensor_list[0].device,
            )

        for i, x in enumerate(tensor_list):
            seq_len = min(x.shape[0], max_seq_len)
            batch_tensor[i, :seq_len] = x[:seq_len]
            mask[i, :seq_len] = True

        return batch_tensor, mask

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

    def forward(self, llm_text=None, llm_images=None, goal=None, attention_mask=None, num_samples=None):
        """
        hidden_states: [B, T, D]
        goal: optional dict:
            goal["goal_position"]: [B, 2]
            goal["goal_yaw"]:      [B, 1]
        attention_mask: optional [B, T]
        num_samples: number of goals to sample during inference
        """
        all_hidden_states = generate_llm_output(llm_text, llm_images, self.llm_config, self.llm_model, self.llm_tokenizer)
        llm_hidden_states, llm_mask = self._bundle_tensors(all_hidden_states, max_seq_len=512)

        b = llm_hidden_states.shape[0]

        feat = self.sequence_encoder(
            llm_hidden_states.float(),
            attention_mask=attention_mask,
        )  # [B, feat_dim]

        prior_mu, prior_logvar = self.split_mu_logvar(self.prior_net(feat))

        if goal is not None:
            goal_vec = torch.cat(
                [goal["goal_position"], goal["goal_yaw"]],
                dim=-1,
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

        # Inference: sample from prior p(z | h)
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