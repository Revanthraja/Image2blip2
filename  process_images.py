import torch
import json
import os
import random
import argparse

from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class PreProcessImages:
    def __init__(
        self,
        config_name,
        config_save_name,
        image_directory,
        clip_frame_data,
        beam_amount,
        prompt_amount,
        min_prompt_length,
        max_prompt_length,
        save_dir,
    ):
        self.prompt_amount = prompt_amount
        self.image_directory = image_directory
        self.clip_frame_data = clip_frame_data
        self.image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

        self.processor = None
        self.blip_model = None
        self.beam_amount = beam_amount
        self.min_length = min_prompt_length
        self.max_length = max_prompt_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir

        self.config_name = config_name
        self.config_save_name = config_save_name

    def build_base_config(self):
        return {
            "name": self.config_name,
            "data": []
        }

    def build_image_config(self, image_path: str):
        return {
            "image_path": image_path,
            "data": []
        }

    def build_image_data(self, prompt: str):
        return {
            "prompt": prompt
        }

    def load_blip(self):
        print("Loading BLIP2")

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        model.to(self.device)

        self.processor = processor
        self.blip_model = model

    def process_blip(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.blip_model.generate(
            **inputs,
            num_beams=self.beam_amount,
            min_length=self.min_length,
            max_length=self.max_length
        )
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True)[0].strip()

        return generated_text

    def save_captions(self, config: dict):
        captions = []
        for image_config in config["data"]:
            for image_data in image_config["data"]:
                captions.append(image_data["prompt"])

        captions_path = f"{self.save_dir}/{self.config_save_name}_captions.txt"
        with open(captions_path, "w") as captions_file:
            captions_file.write("\n".join(captions))

    def process_images(self):
        self.load_blip()
        config = self.build_base_config()

        if not os.path.exists(self.image_directory):
            raise ValueError(f"{self.image_directory} does not exist.")

        for _, _, files in tqdm(
            os.walk(self.image_directory),
            desc=f"Processing images in {self.image_directory}",
        ):
            for image_file in files:
                if image_file.endswith(self.image_extensions):
                    image_path = f"{self.image_directory}/{image_file}"
                    image = Image.open(image_path)

                    image_config = self.build_image_config(image_path)

                    for _ in range(self.prompt_amount):
                        prompt = self.process_blip(image)
                        image_data = self.build_image_data(prompt)

                        image_config["data"].append(image_data)

                    config['data'].append(image_config)

        print(f"Done. Saving train config to {self.save_dir}.")
        self.save_train_config(config)
        self.save_captions(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_name', help="The name of the configuration.", type=str, default='My Config'
    )
    parser.add_argument(
        '--config_save_name',
        help="The name of the config file that's saved.",
        type=str,
        default='my_config',
    )
    parser.add_argument(
        '--image_directory',
        help="The directory where your images are located.",
        type=str,
        default='./images',
    )
    parser.add_argument(
        '--clip_frame_data',
        help="Save the images as clips to HDD/SDD.",
        action='store_true',
        default=False,
    )
    parser.add_argument('--beam_amount', help="Amount for BLIP beam search.", type=int, default=7)
    parser.add_argument('--prompt_amount', help="The amount of prompts per image that is processed.", type=int, default=25)
    parser.add_argument('--min_prompt_length', help="Minimum words required in prompt.", type=int, default=15)
    parser.add_argument('--max_prompt_length', help="Maximum words required in prompt.", type=int, default=30)
    parser.add_argument('--save_dir', help="The directory to save the config to.", type=str, default=f"{os.getcwd()}/train_data")

    args = parser.parse_args()

    processor = PreProcessImages(**vars(args))
    processor.process_images()
