import gradio as gr
import cv2
import numpy as np
import sys
import os

def downsample_image(im):
    return cv2.resize(im, (512, 512))

def get_mask(im):
    mask = np.abs(im["composite"] - im["background"])
    mask[:,:,3] = 255
    binmask = cv2.GaussianBlur(mask, (5, 5), 1)
    binmask[binmask > 0] = 255                              # foreground: white
    binmask[np.all(mask[:, :, :3] == 0, axis=-1), 3] = 0    # background: transparent
    return binmask

def get_object(im):
    obj = im["background"]
    binmask = get_mask(im)
    obj[binmask == 0] = 0
    return obj

# dilate the mask for bg inpainting
def dilate_mask(mask, kernel_size=30):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

