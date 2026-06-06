import torch
from torch import nn

import math


class PositionalEncoder(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim, matrix_dim=3):
        super(PositionalEncoder, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))

        if matrix_dim == 4:
            pe = pe.unsqueeze(0).unsqueeze(0)
        elif matrix_dim == 3:
            pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.matrix_dim = matrix_dim

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)

        if self.matrix_dim == 4:
            seq_len = x.size(2)
            x = x + torch.autograd.Variable(self.pe[:, :, :seq_len, :], requires_grad=False)
        elif self.matrix_dim == 3:
            seq_len = x.size(1)
            x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)

        return x