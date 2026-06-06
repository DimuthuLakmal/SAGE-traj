import pandas as pd
from transformers import DebertaTokenizer, RobertaTokenizer, AutoTokenizer
import pickle

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# # Initialize the tokenizer
# tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")


def tokenize_str(text_arr):
    # Tokenize the text column
    input_ids = []
    attention_mask = []

    for text in text_arr:
        tokenized_text = tokenizer(
            text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
        )

        input_ids.append(tokenized_text["input_ids"].squeeze(0))
        attention_mask.append(tokenized_text["attention_mask"].squeeze(0))

    return input_ids, attention_mask