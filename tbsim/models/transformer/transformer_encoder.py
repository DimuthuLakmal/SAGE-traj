import torch
from torch import nn

from tbsim.models.transformer.encoder_block import EncoderBlock
from tbsim.models.transformer.positional_encoding import PositionalEncoder


class TransformerEncoder(nn.Module):
    def __init__(self,
            dim_model,
            num_heads,
            num_encoder_layers,
            dropout_p,
            max_seq_len,
            agent_hist=True,
            map=True):
        super(TransformerEncoder, self).__init__()

        self.dropout_p = dropout_p
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_encoder_layers
        self.map = map
        self.agent_hist = agent_hist
        out_dim = dim_model

        # embedding and positional encoder
        agent_hist_emb_dim = 64
        self.dim_model = 0
        if agent_hist:
            self.dim_model += agent_hist_emb_dim
        if map:
            self.dim_model += dim_model

        self.agent_hist_emb = nn.Linear(8, agent_hist_emb_dim)
        self.positional_encoder = PositionalEncoder(max_seq_len, self.dim_model)

        # encoder attention blocks
        self.layers = nn.ModuleList(
            [EncoderBlock(
                embed_dim=self.dim_model,
                num_heads=self.num_heads,
                src_dropout=.2,
                ff_dropout=0.2,
                expansion_factor=4
            ) for i in range(self.num_layers)])

        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=self.dim_model, out_channels=self.dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(self.num_layers)])

        # self.final_conv = nn.Conv1d(in_channels=self.max_seq_len - 1, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.fc_out_1 = nn.Linear(31 * dim_model, out_dim * 10)
        self.fc_out_2 = nn.Linear(10 * dim_model, out_dim * 5)
        self.fc_out_3 = nn.Linear(5 * dim_model, out_dim)

    def forward(self, x):
        if self.agent_hist:
            agent_hist_x = x[1]
            x = torch.concat((x[0], self.agent_hist_emb(agent_hist_x)), dim=-1)

        out_e = self.positional_encoder(x)

        for enc_layer, conv_layer in zip(self.layers, self.conv_layers):
            # out_e = conv_layer(out_e.transpose(1, 2)).transpose(1, 2)
            q, k, v = out_e, out_e, out_e
            out_e = enc_layer(q, k, v)  # output of temporal encoder layer

        # out_e = self.final_conv(out_e)

        out_e = out_e.reshape((out_e.shape[0], out_e.shape[1] * out_e.shape[2]))    
        out_e = self.fc_out_1(out_e)
        out_e = self.fc_out_2(out_e)
        out_e = self.fc_out_3(out_e)

        return out_e

        # return torch.squeeze(out_e)
