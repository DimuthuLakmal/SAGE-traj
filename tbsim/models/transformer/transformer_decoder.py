from torch import nn

from tbsim.models.transformer.decoder_block import DecoderBlock
from tbsim.models.transformer.token_embedding import TokenEmbedding


class TransformerDecoder(nn.Module):
    def __init__(self, dim_model, num_heads, num_layers):

        super(TransformerDecoder, self).__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        emb_dim = 16
        input_dim = 2
        out_dim = 2
        # input_dim = configs['input_dim']

        # embedding
        self.embedding = TokenEmbedding(input_dim=input_dim, embed_dim=emb_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    emb_dim=dim_model,
                    n_head=num_heads,
                    src_dropout=0.2,
                    ff_dropout=0.2,
                    cross_attn_dropout=0.2,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(dim_model, out_dim)


    def forward(self, x_noise, enc_x):
        out_d = self.embedding(x_noise)
        # out_d = self.fc_proj(torch.cat([out_d, x_cond, t], dim=-1))
        # out_d = self.fc_proj(out_d + t)
        if(len(out_d.shape) == 2):
            out_d = out_d.unsqueeze(dim=1)

        for idx, layer in enumerate(self.layers):
            out_d = layer(out_d, enc_x)

        return self.fc_out(out_d.squeeze(dim=1))
