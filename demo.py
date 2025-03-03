import gradio as gr
import cv2
import numpy as np
import sys
import os
import subprocess
import torch
import argparse
from omegaconf import OmegaConf
from PIL import Image
from utils.transform import mask_transform, object_transform, match_brightness
from utils.preprocess import downsample_image, get_mask, dilate_mask, get_object

# sys.path.append(os.path.join(os.path.dirname(__file__), "GroundingDINO"))

# from GroundingDINO.seg import inference, load_pil_image

############################### Wonder3D Inference ###############################

# Add Wonder3D to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), "Wonder3D"))

# set device to cuda
device = torch.device('cuda:0')

# for prompt mode
def save_object(im, prompt, folder_path):
    try:
        original = cv2.cvtColor(im["background"], cv2.COLOR_RGBA2BGRA)
        original = downsample_image(original)
        cv2.imwrite(os.path.join(folder_path, "bg/image1.png"), original)
        # pilimg = load_pil_image(os.path.join(folder_path, "bg/image1.png"))
        result = subprocess.run(["bash", "segment.sh"], capture_output=False, text=True)
        # inference(pilimg, prompt, os.path.join(folder_path, "object.png"))
        img = cv2.imread(os.path.join(folder_path, "object.png"))
        img2 = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        cv2.imwrite(os.path.join(folder_path, "object.png"), img)
        mask = (img[:, :, -1] > 0).astype(np.uint8)
        mask[mask > 0] = 255
        mask = dilate_mask(mask)
        mask = downsample_image(mask)
        cv2.imwrite(os.path.join(folder_path, "bg/image1_mask001.png"), mask)
        # return img2
    except Exception as e:
        print(f"Error executing script: {str(e)}")
        return f"Error executing script: {str(e)}"

def trigger_paint(folder_path, output_dir, reference="right"):
    # inpaint the background
    # bg_inpaint = subprocess.run(["bash", "inpaint.sh"], capture_output=False, text=True)

    # script for benchmark_1
    
    # set data path 
    data_path = os.path.join(folder_path, "data")
    
    # set novel view path
    nv_path = os.path.join(folder_path, "gen_data")
    
    for folder in os.listdir(nv_path):
        refer_path = os.path.join(data_path, folder) # input_dir/data/img
        folder_path = os.path.join(nv_path, folder)  # input_dir/gen_data/img
        output_path = os.path.join(output_dir, folder) # input_dir/inpaints_results/img
        if not os.path.isdir(folder_path):
            print(f"{folder_path} is not a directory")
            continue
        
        for angle in os.listdir(folder_path):
            output_path = os.path.join(output_path, angle) # input_dir/inpaints_results/img/angle
            angle_path = os.path.join(folder_path, angle) # input_dir/gen_data/img/angle 
            
            for i in range(0, 3): # loop through 4 samples of same angle image
                # set nv image path
                input_path = os.path.join(angle_path, f"{i}.png")
                print(f'current processing {input_path}')
    
                # results/image1_mask001.png is background image
                bg = cv2.imread(refer_path + "/bg.png")
                bg = cv2.cvtColor(bg, cv2.COLOR_BGRA2RGBA)

                # img will load the generated angle image -> change the path to the generated image
                # img = cv2.imread(folder_path + "/object/mv-enhancement-0/ortho/colors/rgb_000_" + reference + ".png")
                img = cv2.imread(input_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                object_extracted = cv2.bitwise_and(img, img, mask=mask)
                #mask is the rotated object mask
                cv2.imwrite(refer_path + "/genmask_1.png", mask)
                cv2.imwrite(refer_path + "/object_extracted_1.png", object_extracted)
                # cv2.imwrite(folder_path + "/object/mv-enhancement-0/reference.png", mask)
                # cv2.imwrite(folder_path + "/object/mv-enhancement-0/object_extracted.png", object_extracted)

                old_mask = cv2.imread(refer_path + "/mask.png", cv2.IMREAD_GRAYSCALE)
                new_mask = cv2.imread(refer_path + "/genmask_1.png", cv2.IMREAD_GRAYSCALE)
                old_obj = cv2.imread(refer_path + "/object_extracted_1.png")
                new_obj = object_transform(old_mask, old_obj, new_mask)

                cv2.imwrite(refer_path + "/object_transformed.png", new_obj)
                # cv2.imwrite(folder_path + "/object/mv-enhancement-0/object_transformed.png", new_obj)
                mask2 = mask_transform(old_mask, new_mask)
                cv2.imwrite(refer_path + "/reference_transformed.png", mask2)
                new_obj = cv2.imread(refer_path + "/object_transformed.png")
                old_obj = cv2.imread(refer_path + "/front_img.png")
                new_obj = match_brightness(old_obj, new_obj)
    
                new_obj = cv2.cvtColor(new_obj, cv2.COLOR_BGRA2RGBA)
                mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGRA2RGBA)
                # do erosion to remove the white border
                kernel = np.ones((5,5), np.uint8)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_ERODE, kernel)
                mask2 = cv2.cvtColor(mask2, cv2.COLOR_RGBA2BGRA)

                # paste the object onto the background
                bg[mask2 > 0] = new_obj[mask2 > 0]
                bg = cv2.cvtColor(bg, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(output_path + "/output_{i}.png", bg)
                print(f"{output_path} - output_{i}.png saved")
                # output = cv2.cvtColor(bg, cv2.COLOR_BGRA2RGBA)
                # return output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/rotate')
    parser.add_argument('--output_dir', type=str, default='data/rotate')
    parser.add_argument('--reference', type=str, default='behind')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    trigger_paint(args.input_dir, args.output_dir, args.reference)