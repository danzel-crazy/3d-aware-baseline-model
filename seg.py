import math

from PIL import Image, ImageOps
from transformers import logging

from mask_extractor import ExternalMaskExtractor
import os

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

    #mask the object to white color
    
    # Create an all-white image
    print(image_pil.size)
    white_image = Image.new("RGB", image_pil.size, (255, 255, 255))  # White image

    # Apply the mask: Keep white where the mask is white, black otherwise
    white_edited_image = Image.composite(white_image, Image.new("RGB", image_pil.size, (0, 0, 0)), mask_pil)

    return edited_image, white_edited_image

# Different edit instruction and path
# image_path = "/tmp2/danzel/mask_extractor/test_images/rotate_mug.jpg"
# edit_instruction = 'move the white mug to the right'


# image_path = "/tmp2/danzel/mask_extractor/test_images/rotate_fucet.png"
# edit_instruction = 'move the blue fucet to the right'

image_path = "/tmp2/danzel/mask_extractor/test_images/move_cube.jpg"
edit_instruction = 'move the blue cube to the right'

image_name = os.path.basename(image_path).split('.')[0]
print(f"Image name: {image_name}")

image = load_pil_image(image_path, resolution=512)
# image = load_pil_image(image_path, resolution=256)
image.show()

#create eobject mask
edited_image, white_edited_image = inference(image, edit_instruction)
# edited_image.show()
save_path = "/tmp2/danzel/mask_extractor/test_images/" + image_name + "_edited.png"
edited_image.save(save_path, format="PNG", optimize=True)

#create white object mask
white_save_path = "/tmp2/danzel/mask_extractor/test_images/" + image_name + "_white.png"
white_edited_image.save(white_save_path, format="PNG", optimize=True)

