import pandas as pd
import base64

from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from trl import SFTTrainer
from transformers.image_utils import load_image
from tqdm import tqdm


system_message = """You are a Vision Language Model specialized in interpreting map images and coordinate sequences. 
You are provided with a map image and coordinate sequences of vehicles in a tabular format.
The map is a 224*224 three color images of a road segment from bird eye view. Yellow lines represent white lane lines in roads. 
Blue color area represent pavements and red area represents the road and sometimes, parking areas where vehicle can move. 
The tabular data shows the movements of vehicles on this road segment captured on 8 time steps in a fixed frequency. Each row has both x and y coordinates recorded in each time step, in the format of x1,y1,x2,y2,...,x8,y8|. | represents the line break. 
The first line of the tabular data has the column names and second line has the ego vehicle movement whose future movements are going to be predicted based on these historical observations and the road segment. 
The remaining rows have the movements of other vehicles on the road. 
In some time steps, both x and y values are 0, and they are noises. 
Don't consider them in the analysis. When overlaying the coordinates on the map to understand movements, make sure that (0,0) coordinate starts from the top left corner. 
Considering these information, You need to generate a text describing the movement of the vehicles on the road, context of covering constraints and prominent road features like intersections, bends that it has to be aware of when generating future movements. 
Text must have less than or equal 512 tokens. Give the output as few sentences. 
Don't repeat what I said in the instructions. 
Be more specific to the given map and given vehicle movements. 
Don't give more generic information. 
For example, with regards to the movements, you could say 'Ego vehicles turns right from the intersection into the outer lane.'. With regards to the context, 'There's a T intersection towards the center left of the map'. 
These examples may not relevant to the given scenario. 
They're just examples showing the format I need in the output. Avoid additional explanation unless absolutely necessary."""


def format_data(image_file, data):
    image = load_image(image_file)
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": data,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": None}],
        },
    ]


def generate_text_from_sample(model, processor, sample, max_new_tokens=2048, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[:2], tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )

    # print("Expected output ", sample[2])

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=4096)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

def retrieve_llm_data(maps_root_dir, batch_hist_pos):
    # Process each pair of files

    responses = []

    for i, hist_pos in enumerate(batch_hist_pos):
        hist_pos = hist_pos.permute(1, 0, 2)

        image_file = f"{maps_root_dir}/maps_map9_guide_{i}.png"
        hist_pos = process_location_data(hist_pos)

        request = format_data(image_file, hist_pos)
        output = generate_text_from_sample(model, processor, request)
        output = output.split("<|im_start|>assistant\n")[1].split("\n<|im_end|>")[0]
        responses.append(output)

    return responses


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def process_location_data(hist_pos_data):
    lines = "x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8|"
    for vehicle_data in hist_pos_data:
        vehicle_data = [f"{x[0].item()},{x[1].item()}" for i, x in enumerate(vehicle_data) if i % 4 == 0]
        line = ",".join(vehicle_data)
        lines += line + "|"

    return lines


def tokenize_data(data):
    return data.split("|")
    