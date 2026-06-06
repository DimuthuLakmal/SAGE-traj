from torch import nn

from tbsim.models.transformer.cross_attention import CrossAttentionLayer
from tbsim.models.transformer.multi_head_attention import MultiHeadAttention
from tbsim.models.transformer.position_wise_feed_forward import PositionWiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, n_head, src_dropout, ff_dropout, cross_attn_dropout,):
        super(DecoderBlock, self).__init__()

        emb_dim = emb_dim
        n_heads = n_head
        expansion_factor = 4
        cross_attn_dropout = cross_attn_dropout
        src_dropout = src_dropout
        ff_dropout = ff_dropout

        self.self_attention = MultiHeadAttention(emb_dim, n_heads, mask=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(src_dropout)

        self.cross_attn_layer = CrossAttentionLayer(emb_dim, n_heads, dropout=cross_attn_dropout)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.feed_forward = PositionWiseFeedForward(emb_dim, expansion_factor * emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(ff_dropout)

    def forward(self, x, enc_x):
        # self attention
        attention = self.self_attention(x, x, x)  # 32x10x512
        x = self.norm1(x + self.dropout1(attention))

        # cross attention
        cross_attn = self.cross_attn_layer(x, enc_x, enc_x)
        x = self.norm2(x + cross_attn)

        # positionwise ffn
        ff_output = self.feed_forward(x)
        out = self.norm3(x + self.dropout2(ff_output))

        return out
