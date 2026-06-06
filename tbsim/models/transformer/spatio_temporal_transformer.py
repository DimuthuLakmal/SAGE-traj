import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers import RobertaModel, DebertaModel
from transformers import AutoModel

from peft import LoraConfig, get_peft_model

# from tbsim.models.deberta.deberta import DebertaWithSingleOutput
from tbsim.models.diffuser_helpers import SinusoidalPosEmb
from tbsim.models.transformer.graph_transformer_encoder import GraphTransformerEncoder
from tbsim.models.transformer.transformer_encoder import TransformerEncoder

from tbsim.utils.internvl_utils import get_item, generate_llm_output
from tbsim.models.internvl.train.dataset import build_transform
from tbsim.models.transformer.llm_attention_pooling import AttentionPooling

from torchvision.utils import save_image


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, encoder_config, decoder_config, llm_model, llm_tokenizer, llm_config):
        super(SpatioTemporalTransformer, self).__init__()

        self.llm_proj = nn.Linear(256, 64)
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
        self.llm_attn_pool = AttentionPooling(hidden_dim=self.llm_model.language_model.config.hidden_size)
        self.llm_proj = nn.Linear(self.llm_model.language_model.config.hidden_size, 256)

        # self.graph_encoder = GraphTransformerEncoder(
        #     dim_model=64,
        #     num_heads=encoder_config['num_heads'],
        #     num_encoder_layers=encoder_config['num_layers'],
        #     dropout_p=0.2,
        #     max_seq_len=encoder_config['max_seq_len']).to('cuda')


        dim_model = encoder_config['dim_model']
        # t_dim = 256
        cond_dim = 256 + 64
        # proj_input_dim = cond_dim + t_dim
        self.fc_enc_proj = nn.Linear(in_features=cond_dim, out_features=128)

    
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

    def forward(self, aux_info):

        all_hidden_states = generate_llm_output(aux_info['llm_text'], aux_info['llm_images'], self.llm_config, self.llm_model, self.llm_tokenizer)

        llm_hidden_states, llm_mask = self._bundle_tensors(all_hidden_states, max_seq_len=512)
        llm_enc_out = self.llm_attn_pool(llm_hidden_states.float(), llm_mask)
        llm_enc_out = self.llm_proj(llm_enc_out[0])

        # graph_enc_out = self.graph_encoder(aux_info)
        # image_enc_out = self.image_encoder(aux_info['map_global_feat_hist'])

        # enc_out = torch.cat([graph_enc_out, aux_info['map_global_feat_hist']], dim=-1)

        # dec_out = self.decoder(x_noise, enc_out)
        # enc_out = enc_out.reshape((enc_out.shape[0], enc_out.shape[1] * enc_out.shape[2]))
        # enc_out = self.fc_enc_proj(enc_out)

        return llm_enc_out

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
