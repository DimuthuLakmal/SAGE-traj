import torch
from tbsim.models.internvl.train.dataset import dynamic_preprocess
from PIL import Image

# LLM image processing
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tbsim.models.internvl.train.constants import (IMAGENET_MEAN, IMAGENET_STD)
transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])


import json
import os

from transformers import AutoTokenizer
from tbsim.models.internvl.model.internvl_chat import InternVLChatModel
from tbsim.models.internvl.model.goal_predictor.mllm_spatial_goal_predictor import MLLMGoalPredictor


def get_pixel_values(input_images, config):
        images, num_tiles = [], []
        num_image = len(input_images)

        for image in input_images:
            # image_path = config.image_root + image_path
            # image = Image.open(image_path).convert("RGB")

            if config['dynamic_image_size']:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, image_size=config['input_size'],
                                           use_thumbnail=config['use_thumbnail'],
                                           max_num=config['max_num'])
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)

        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return pixel_values


def load_llm_checkpoint(
    checkpoint_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    system_message: str = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    load_tokenizer: bool = True,
    use_fast_tokenizer: bool = False,
):
    """
    Load the InternVL component of a component-checkpoint saved by
    TargetSFTTrainer (see internvl/train/target_trainer.py).

    Mode (train/eval) is left to the caller: this does not call .eval().
    """
    internvl_dir = os.path.join(checkpoint_dir, 'internvl_model')

    internvl_model = InternVLChatModel.from_pretrained(
        internvl_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        system_message=system_message,
    )

    if not load_tokenizer:
        return internvl_model

    tokenizer = AutoTokenizer.from_pretrained(
        internvl_dir,
        trust_remote_code=True,
        use_fast=use_fast_tokenizer,
    )
    return internvl_model, tokenizer


def load_goal_predictor_checkpoint(
    checkpoint_dir: str,
    internvl_model: InternVLChatModel,
    tokenizer: AutoTokenizer,
):
    """
    Reconstruct MLLMGoalPredictor from the predictor-only weights and config
    saved by TargetSFTTrainer._save alongside the InternVL checkpoint.
    """
    predictor_dir = os.path.join(checkpoint_dir, 'goal_predictor')
    predictor_config_path = os.path.join(predictor_dir, 'goal_predictor_config.json')
    predictor_weights_path = os.path.join(predictor_dir, 'goal_predictor_weights.pt')

    for path in (predictor_config_path, predictor_weights_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Required checkpoint component was not found: {path}')

    with open(predictor_config_path, 'r', encoding='utf-8') as file:
        predictor_config = json.load(file)

    goal_model = MLLMGoalPredictor(
        **predictor_config,
        llm_model=internvl_model,
        llm_tokenizer=tokenizer,
    )

    try:
        predictor_state_dict = torch.load(predictor_weights_path, weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch releases.
        predictor_state_dict = torch.load(predictor_weights_path)

    incompatible = goal_model.load_state_dict(predictor_state_dict, strict=False)

    # llm_model.* is deliberately absent from predictor_weights.pt.
    invalid_missing_keys = [
        key for key in incompatible.missing_keys if not key.startswith('llm_model.')
    ]
    if invalid_missing_keys:
        raise RuntimeError(
            'Predictor checkpoint is missing non-InternVL parameters:\n'
            + '\n'.join(invalid_missing_keys)
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            'Unexpected predictor checkpoint parameters:\n'
            + '\n'.join(incompatible.unexpected_keys)
        )

    return goal_model

# def get_item(images, text, config):
#     pixel_values = multi_modal_multi_image_get_item(images, config)
#     human_msg = 't4:<image>\n t8:<image>\n t12:<image>\n t16:<image>\nx1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8|' + text

#     return pixel_values, human_msg


# def generate_llm_output(questions, pixel_values, generation_config, llm_model, llm_tokenizer):
#     pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(llm_model.device)
#     pv_shape = pixel_values.shape
#     pixel_values_cat = pixel_values.reshape(pv_shape[0] * pv_shape[1], pv_shape[2], pv_shape[3], pv_shape[4])
#     with torch.inference_mode():
#         preds, b_hidden_states, b_seq_end_lengths = llm_model.batch_chat(
#             tokenizer=llm_tokenizer,
#             pixel_values=pixel_values_cat,
#             questions=questions,
#             generation_config=generation_config,
#             # save_logits=True
#         )

#         b_last_hidden_states = []
#         for i in range(len(questions)):
#             q_hidden_states = b_hidden_states[:b_seq_end_lengths[i]]
#             q_last_hidden_states = torch.stack([w_hidden_states[-1][i, -1, :] for w_hidden_states in q_hidden_states])
#             b_last_hidden_states.append(q_last_hidden_states)

#     return b_last_hidden_states
