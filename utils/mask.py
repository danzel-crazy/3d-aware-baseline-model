import gradio as gr
import time
import cv2
import numpy as np

def get_mask(im):
    mask = np.abs(im["composite"] - im["background"])
    mask[:,:,3] = 255
    binmask = mask[:,:,0] + mask[:,:,1] + mask[:,:,2]
    binmask = gaussian_filter(binmask, 1)
    binmask[binmask > 0] = 255

    # binmask = binmask.astype(np.uint8) * 255
    return binmask

def gaussian_filter(im, sigma=1):
    # apply gaussian filter to the image
    return cv2.GaussianBlur(im, (5, 5), sigma)

def downsample_image(im):
    return cv2.resize(im, (512, 512))

def save_results(im):
    original = cv2.cvtColor(im["background"], cv2.COLOR_RGBA2BGRA)
    original = downsample_image(original)
    cv2.imwrite("original.png", original)
    maskedimg = cv2.cvtColor(im["composite"], cv2.COLOR_RGBA2BGRA)
    maskedimg = downsample_image(maskedimg)
    cv2.imwrite("mask.png", maskedimg)
    binmask = get_mask(im)
    binmask = downsample_image(binmask)
    cv2.imwrite("mask_binary.png", binmask)

def update_preview(image):
    binmask = get_mask(image)
    return binmask

# Create a Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        im = gr.ImageEditor(
            type="numpy",
        )
        im_preview = gr.Image(
            type="numpy",
            label="Mask Preview"
        )
    im.change(update_preview, inputs=im, outputs=im_preview)
    
    save_btn = gr.Button("Save Mask")
    save_btn.click(fn=save_results, inputs=im)

if __name__ == "__main__":
    demo.launch()



