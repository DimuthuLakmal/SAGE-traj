import torch
from torch import nn

from tbsim.models.transformer.encoder_block import EncoderBlock
from tbsim.models.transformer.positional_encoding import PositionalEncoder


class UnetTransformerEncoder(nn.Module):
    def __init__(self,
            dim_model,
            num_heads,
            num_encoder_layers,
            dropout_p,
            max_seq_len):

        super(UnetTransformerEncoder, self).__init__()

        out_dim = 2

        # embedding and positional encoder
        self.action_emb = nn.Linear(2, dim_model)
        self.positional_encoder = PositionalEncoder(max_seq_len, dim_model)

        # encoder attention blocks
        self.layers = nn.ModuleList(
            [EncoderBlock(
                embed_dim=dim_model,
                num_heads=num_heads,
                src_dropout=.1,
                ff_dropout=0.1,
                expansion_factor=4,
                mask=True
            ) for i in range(num_encoder_layers)])

        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(num_encoder_layers)])

        self.fc_out = nn.Linear(dim_model, out_dim)

    def forward(self, x):
        out_e = self.positional_encoder(self.action_emb(x))

        for enc_layer, conv_layer in zip(self.layers, self.conv_layers):
            out_e = conv_layer(out_e.transpose(1, 2)).transpose(1, 2)
            q, k, v = out_e, out_e, out_e
            out_e = enc_layer(q, k, v)  # output of temporal encoder layer
  
        return self.fc_out(out_e)
