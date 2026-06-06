import torch
from transformers import DebertaModel

# Define a custom model with pooling for single output
class DebertaWithSingleOutput(torch.nn.Module):
    def __init__(self):
        super(DebertaWithSingleOutput, self).__init__()
        self.deberta = DebertaModel.from_pretrained("microsoft/deberta-base")
        for param in self.deberta.parameters():
            param.requires_grad = False
        self.pooler = torch.nn.Linear(self.deberta.config.hidden_size, 64)  # Reduce to desired dimension

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token (first token) as a single output
        cls_representation = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.pooler(cls_representation)
        return pooled_output
