import os
from time import sleep
import random

import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting


def generate(image1, text1, generation_config, safety_settings):
    vertexai.init(project="dc-digital-twin", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-flash-002",
    )
    responses = model.generate_content(
        [image1, text1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    text_response = ""
    for response in responses:
        text_response += response.text
    print(f"Response: {text_response}")

    # sleep(1)
    return text_response

# Process each pair of files
def retrieve_llm_data(maps_root_dir, batch_hist_pos):
    # Process each pair of files

    responses = []

    for i, hist_pos in enumerate(batch_hist_pos):
        hist_pos = hist_pos.permute(1, 0, 2)

        encoded_image = encode_image(f"{maps_root_dir}/maps_{i}.png")
        hist_pos = process_location_data(hist_pos)

        # document1 = Part.from_data(
        #     mime_type="text/plain",
        #     data=csv_summary,
        # )
        text1 = """This map is a 224*224 three color images of a road segment from bird eye view. Yellow lines represent white lane lines in roads. Blue color area represent pavements and red area represents the road and sometimes, parking areas where vehicle can move. The csv data shows the movements of vehicles on this road segment captured on 8 time steps in a fixed frequency. Each row has both x and y coordinates recorded in each time step, in the format of x1,y1,x2,y2,...,x8,y8|. | represents the line break. The first line of the csv data has the column names and second line has the ego vehicle movement whose future movements are going to be predicted based on these historical observations and the road segment. The remaining rows have the movements of other vehicles on the road. In some time steps, both x and y values are 0, and they are noises. Don't consider them in the analysis. When overlaying the coordinates on the map to understand movements, make sure that (0,0) coordinate starts from the top left corner. Considering these information, I want to generate a text describing the movement of the vehicles on the road, context of covering constraints and prominent road features like intersections, bends that it has to be aware of when generating future movements. Text must have less than or equal 512 tokens. Give the output as few sentences. Don't repeat what I said in the instructions. Be more specific to the given map and given vehicle movements. Don't give more generic information. For example, with regards to the movements, you could say 'Ego vehicles turns right from the intersection into the outer lane.'. With regards to the context, 'There's a T intersection towards the center left of the map'. These examples may not relevant to the given scenario. They're just examples showing the format I need in the output. csv data are: """
        text1 += hist_pos

        generation_config = {
            "max_output_tokens": 512,
            "temperature": 0,
            "top_p": 0.95,
        }

        safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
        ]

        base_delay = 1
        retry_count = 0
        max_retries = 4
        while retry_count < max_retries:
            try:
                text_response = generate(encoded_image, text1, generation_config, safety_settings)
                responses.append(text_response)
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                wait_time = base_delay * (2 ** retry_count) + random.uniform(0, 0.5)  # Exponential backoff with jitter
                print(f"Attempt {retry_count} failed: {e}. Retrying in {wait_time:.2f} seconds...")

                if("it has an empty text parameter" in str(e)):
                    responses.append("It's not a valid input. So the system fails to analyse the input correctly")
                    break

                sleep(wait_time)


    return responses


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def process_location_data(hist_pos_data):
    lines = ""
    for vehicle_data in hist_pos_data:
        vehicle_data = [f"{x[0].item()},{x[1].item()}" for i, x in enumerate(vehicle_data) if i % 4 == 0]
        line = ",".join(vehicle_data)
        lines += line + "|"

    return lines


def tokenize_data(data):
    return data.split("|")