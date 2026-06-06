import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.score = nn.Linear(hidden_dim, num_heads)

    def forward(self, x, attention_mask=None):
        # x: [B, T, D]
        scores = self.score(x)  # [B, T, H]
        scores = scores.permute(0, 2, 1)  # [B, H, T]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)  # [B, 1, T]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # [B, H, T]

        # weighted sum for each head
        pooled = torch.einsum('bht,btd->bhd', attn, x)  # [B, H, D]

        # flatten heads
        pooled = pooled.reshape(x.size(0), -1)  # [B, H*D]
        return pooled, attn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x, attention_mask=None):
        """
        x: [B, T, D]
        attention_mask: [B, T] with 1 for valid tokens, 0 for padding
        """
        # Compute token scores
        scores = self.score(x).squeeze(-1)   # [B, T]

        # Mask padding tokens
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Normalize over sequence dimension
        attn_weights = F.softmax(scores, dim=-1)   # [B, T]

        # Weighted sum of token embeddings
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, D]

        return pooled, attn_weights