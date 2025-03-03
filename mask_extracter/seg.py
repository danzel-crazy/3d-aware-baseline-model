import math

from PIL import Image, ImageOps
from transformers import logging

from mask_extractor import ExternalMaskExtractor
import os
from os.path import join, exists, basename, splitext, dirname
import glob
import json
from collections import OrderedDict

logging.set_verbosity_error()


def load_pil_image(image_path, resolution=512):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    return image


device = 'cuda:0'
mask_extractor = ExternalMaskExtractor(device=device)


def inference(image_pil, instruction):
    external_mask_pil, chosen_noun_phrase, clip_scores_dict = mask_extractor.get_external_mask(image_pil, instruction)
    mask_pil = external_mask_pil.resize(image_pil.size)
    mask_pil = mask_pil.convert("L")
    edited_image = Image.composite(image_pil, Image.new("RGB", image_pil.size, (0, 0, 0)), mask_pil)

    # Ensure the output images are in RGB format
    edited_image = edited_image.convert("RGB")

    # Create an all-white image
    white_image = Image.new("RGB", image_pil.size, (255, 255, 255))  # White image
    white_edited_image = Image.composite(white_image, Image.new("RGB", image_pil.size, (0, 0, 0)), mask_pil)
    white_edited_image = white_edited_image.convert("RGB")

    return edited_image, white_edited_image


def check_image_mode(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        print(f"Warning: {image_path} is not in RGB mode (Detected: {image.mode})")
    else:
        print(f"Success: {image_path} is in RGB mode.")


def seg(input_dir, test_set_path):
    print(input_dir)
    
    # load the test set info
    with open(test_set_path, 'r') as f:
        dataset_names = json.load(f, object_pairs_hook=OrderedDict)
        print(dataset_names)
    
    incomplete_samples = []
    incomplete_json = []
    for sample_name in dataset_names.keys():
        required_fnames = ['input.png', 'mask.png', 'transforms.json']
        missing_files = [fname for fname in required_fnames if not exists(join(input_dir, sample_name, fname))]
        if missing_files:
            print(f"Skipping sample {sample_name}, since it is missing the following required input files: {missing_files}.")
            if 'mask.png' in missing_files:
                incomplete_samples.append(sample_name)
            if 'transforms.json' in missing_files:
                incomplete_json.append(sample_name)
    
    for folder_name in incomplete_samples:
        full_path = os.path.join(input_dir, folder_name)
        print(full_path)  # This prints the full path to the folder
        input_image_path = os.path.join(full_path, 'input.png')
        if exists(input_image_path):
            image_pil = load_pil_image(input_image_path)
            edit_instruction = 'move the ' + folder_name + ' to the right'  # Replace with your actual instruction
            edited_image, white_edited_image = inference(image_pil, edit_instruction)
            edited_image_path = os.path.join(full_path, 'edited_image.png')
            white_edited_image_path = os.path.join(full_path, 'mask.png')
            edited_image.save(edited_image_path, format="PNG", optimize=True)
            white_edited_image.save(white_edited_image_path, format="PNG", optimize=True)
            check_image_mode(edited_image_path)
            check_image_mode(white_edited_image_path)
        else:
            print(f"input.png not found in {full_path}")
            
    for folder_name in incomplete_json:
        full_path = os.path.join(input_dir, folder_name)
        print(full_path)  # This prints the full path to the folder
        transforms_path = os.path.join(full_path, 'transforms.json')
        if not exists(transforms_path):
            default_transforms_path = os.path.join(input_dir, 'default.json')
            if exists(default_transforms_path):
                with open(default_transforms_path, 'r') as f:
                    default_transforms = json.load(f)
            with open(transforms_path, 'w') as f:
                json.dump(default_transforms, f, indent=4)
    
    print(incomplete_samples)


if __name__ == '__main__':
    image_path = "/tmp2/danzel/zero123/data/rotate/rotate_faucet/input.png"
    edit_instruction = 'move the blue faucet to the right'

    image_name = os.path.basename(image_path).split('.')[0]
    print(f"Image name: {image_name}")

    image = load_pil_image(image_path, resolution=512)
    image.show()

    #create object mask
    edited_image, white_edited_image = inference(image, edit_instruction)
    save_path = f"/tmp2/danzel/zero123/data/rotate/rotate_faucet/{image_name}_edited.png"
    white_save_path = f"/tmp2/danzel/zero123/data/rotate/rotate_faucet/{image_name}_white.png"
    edited_image.save(save_path, format="PNG", optimize=True)
    white_edited_image.save(white_save_path, format="PNG", optimize=True)
    
    # Check if saved images are in RGB mode
    check_image_mode(save_path)
    check_image_mode(white_save_path)
