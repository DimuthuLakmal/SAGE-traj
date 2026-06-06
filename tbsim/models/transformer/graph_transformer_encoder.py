import torch
from torch import nn

from tbsim.models.gcn.gcn_fixed_w import GCNConv_Fixed_W
from tbsim.models.transformer.encoder_block import EncoderBlock
from tbsim.models.transformer.positional_encoding import PositionalEncoder


class GraphTransformerEncoder(nn.Module):
    def __init__(self,
            dim_model,
            num_heads,
            num_encoder_layers,
            dropout_p,
            max_seq_len):
        super(GraphTransformerEncoder, self).__init__()

        self.dropout_p = dropout_p
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_encoder_layers
        out_dim = dim_model
        dim_model = 32

        # embedding and positional encoder
        self.hist_emb = nn.Linear(5, dim_model)
        self.positional_encoder = PositionalEncoder(max_seq_len, dim_model, matrix_dim=4)
        self.positional_encoder2 = PositionalEncoder(max_seq_len, dim_model, matrix_dim=4)

        # encoder attention blocks
        self.layers = nn.ModuleList(
            [EncoderBlock(
                embed_dim=dim_model,
                num_heads=self.num_heads,
                src_dropout=.1,
                ff_dropout=0.2,
                expansion_factor=4,
                mask=False
            ) for i in range(self.num_layers)])

        # encoder attention blocks
        self.layers_temp = nn.ModuleList(
            [EncoderBlock(
                embed_dim=dim_model,
                num_heads=2,
                src_dropout=.1,
                ff_dropout=0.2,
                expansion_factor=4,
            ) for i in range(self.num_layers)])

        self.conv_q_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(self.num_layers)])

        self.conv_k_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(self.num_layers)])

        self.conv_q_layers_temp = nn.ModuleList(
            [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(self.num_layers)])

        self.conv_k_layers_temp = nn.ModuleList(
            [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(self.num_layers)])

        self.fc_out = nn.Linear(dim_model, out_dim)

        self.gcn = GCNConv_Fixed_W(
            in_channels=dim_model,
            out_channels=dim_model,
            improved=False,
            cached=False,
            normalize=False,
            add_self_loops=False
        )

        self.batch_norm = nn.BatchNorm2d(1)

        self.lin_graph = nn.Linear(1, dim_model)

        self.fc_enc_proj_1 = nn.Linear(in_features=31 * dim_model, out_features=dim_model*15)
        self.fc_enc_proj_2 = nn.Linear(in_features=15 * dim_model, out_features=out_dim)
        # self.fc_enc_proj_3 = nn.Linear(in_features=5 * dim_model, out_features=dim_model)

        # self.final_conv = nn.Conv1d(in_channels=self.max_seq_len, out_channels=1, kernel_size=3, stride=1, padding=1)

    def _get_edge_index(self, n_nodes):
        return [[x for x in range(n_nodes)], [0] * n_nodes]

    def forward(self, x):
        edge_weights = x['edge_weight']
        edge_idx = x['edge_index']
        x = x['all_hist_feat']
        x = self.hist_emb(x)
        out_e = self.positional_encoder(x)
        out_e_shp = out_e.shape

        for enc_layer, conv_q, conv_k in zip(self.layers, self.conv_q_layers, self.conv_k_layers):
             # output of temporal encoder layer
            out_e = out_e.view(-1, out_e_shp[2], out_e_shp[3])
            out_transposed = out_e.transpose(2, 1)
            q = conv_q(out_transposed).transpose(2, 1)
            k = conv_k(out_transposed).transpose(2, 1)
            v = out_e

            q = q.reshape(out_e_shp[0], out_e_shp[1], out_e_shp[2], out_e_shp[3])
            v = v.reshape(out_e_shp[0], out_e_shp[1], out_e_shp[2], out_e_shp[3])
            k = k.reshape(out_e_shp[0], out_e_shp[1], out_e_shp[2], out_e_shp[3])
            out_e = enc_layer(q, k, v)

        graph_w = out_e.permute(0, 2, 1, 3)
        graph_x = x.permute(0, 2, 1, 3)
        b, t, n, f = graph_x.shape
        graph_x = graph_x.reshape(b, t * n, f)
        graph_w = graph_w.reshape(b, t * n, f)

        graph_out = []
        for w_batch, x_batch, weight_batch, edge_idx_batch in zip(graph_w, graph_x, edge_weights, edge_idx):
            x_batch_out = self.gcn(w_batch, x_batch, edge_idx_batch, weight_batch)
            graph_out.append(x_batch_out.reshape(t, n, -1))

        graph_out = torch.stack(graph_out).permute(0, 3, 2, 1)  # B, F, N, T
        graph_out = self.batch_norm(graph_out).permute(0, 2, 3, 1)

        # Temporal Transoformer
        graph_out = self.lin_graph(graph_out)
        graph_out = self.positional_encoder2(graph_out)
        for enc_layer, conv_q, conv_k in zip(self.layers_temp, self.conv_q_layers_temp, self.conv_k_layers_temp):
            graph_out = graph_out.view(-1, out_e_shp[2], out_e_shp[3])
            out_transposed = graph_out.transpose(2, 1)
            q = conv_q(out_transposed).transpose(2, 1)
            k = conv_k(out_transposed).transpose(2, 1)
            v = graph_out

            q = q.reshape(out_e_shp[0], out_e_shp[1], out_e_shp[2], out_e_shp[3])
            v = v.reshape(out_e_shp[0], out_e_shp[1], out_e_shp[2], out_e_shp[3])
            k = k.reshape(out_e_shp[0], out_e_shp[1], out_e_shp[2], out_e_shp[3])
            graph_out = enc_layer(q, k, v)  # output of temporal encoder layer

        graph_out = graph_out[:, 0].reshape((graph_out.shape[0], graph_out.shape[2] * graph_out.shape[3]))
        graph_out = self.fc_enc_proj_1(graph_out)
        graph_out = self.fc_enc_proj_2(graph_out)
        # graph_out = self.fc_enc_proj_3(graph_out)

        return graph_out