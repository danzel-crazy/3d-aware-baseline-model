from os.path import join, exists
from os import makedirs
import json
from collections import OrderedDict

import torch
from diffhandles import DiffusionHandles

from remove_foreground import remove_foreground
from estimate_depth import estimate_depth
from generate_results_webpage import generate_results_webpage
from utils import crop_and_resize, load_image, load_depth, save_image


def test_diffusion_handles(test_set_path:str, input_dir:str, output_dir:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # TEMP!
    # # need to:
    # # export CUBLAS_WORKSPACE_CONFIG=:4096:8
    # torch.use_deterministic_algorithms(True)
    # torch.manual_seed(76523524)
    # import random
    # random.seed(634566435)
    # import numpy as np
    # np.random.seed(54396745)
    # # TEMP!

    # load the test set info
    # (use an OrderDict just so the samples are always in the same order)
    with open(test_set_path, 'r') as f:
        test_set_info = json.load(f, object_pairs_hook=OrderedDict)

    # check samples for completeness
    incomplete_samples = []
    foreground_removal_samples = []
    depth_estimation_samples = []
    for sample_name, sample_info in test_set_info.items():
        if not exists(join(input_dir, sample_name, 'input.png')) or not exists(join(input_dir, sample_name, 'mask.png')):
            print(f"Skipping sample {sample_name}, since it is missing input.png or mask.png.")
            incomplete_samples.append(sample_name)
            continue
        if not exists(join(input_dir, sample_name, 'bg_depth.exr')) and not exists(join(input_dir, sample_name, 'bg.png')):
            foreground_removal_samples.append(sample_name)
        if not exists(join(input_dir, sample_name, 'depth.exr')) or not exists(join(input_dir, sample_name, 'bg_depth.exr')):
            depth_estimation_samples.append(sample_name)

    # remove incomplete samples
    if len(incomplete_samples) > 0:
        for sample_name in incomplete_samples:
            test_set_info.pop(sample_name)

    # estimate missing background images (with removed foreground object)
    if len(foreground_removal_samples) > 0:
        remove_foreground(
            input_image_paths=[join(input_dir, sample_name, 'input.png') for sample_name in foreground_removal_samples],
            foreground_mask_paths=[join(input_dir, sample_name, 'mask.png') for sample_name in foreground_removal_samples],
            output_paths=[join(input_dir, sample_name, 'bg.png') for sample_name in foreground_removal_samples])

    # estimate missing depths
    if len(depth_estimation_samples) > 0:
        estimate_depth(
            input_image_paths=[
                join(input_dir, sample_name, 'input.png') for sample_name in depth_estimation_samples]+[
                join(input_dir, sample_name, 'bg.png') for sample_name in depth_estimation_samples],
            output_paths=[
                join(input_dir, sample_name, 'depth.exr') for sample_name in depth_estimation_samples]+[
                join(input_dir, sample_name, 'bg_depth.exr') for sample_name in depth_estimation_samples]
            )

    # iterate over test set samples
    for sample_name, sample_info in test_set_info.items():
        
        prompt = sample_info['prompt']
        transforms = sample_info['transforms']

        if 'config_path' in sample_info:
            config_path = sample_info['config_path']
        else:
            config_path = None

        diff_handles = DiffusionHandles(conf_path=config_path)
        diff_handles.to(device)

        makedirs(join(output_dir, sample_name), exist_ok=True)

        # load inputs for the sample
        img, fg_mask, depth, bg_depth = load_diffhandles_inputs(
            sample_dir=join(input_dir, sample_name), img_res=diff_handles.img_res, device=device)
        
        # save inputs for visualization to results directory
        save_image(img[0], join(output_dir, sample_name, 'input.png'))
        save_image(fg_mask[0], join(output_dir, sample_name, 'mask.png'))
        save_image((depth/depth.max())[0], join(output_dir, sample_name, 'depth.png'))
        save_image((bg_depth/bg_depth.max())[0], join(output_dir, sample_name, 'bg_depth.png'))

        # set the foreground object to get inverted null text, noise, and intermediate activations to use as guidance
        bg_depth, inverted_null_text, inverted_noise, activations, activations2, activations3, latent_image = diff_handles.set_foreground(
            img=img, depth=depth, prompt=prompt, fg_mask=fg_mask, bg_depth=bg_depth)

        # save image reconstructed from inversion
        with torch.no_grad():
            latent_image = 1 / 0.18215 * latent_image.detach()
            recon_image = diff_handles.diffuser.vae.decode(latent_image)['sample']
            recon_image = (recon_image + 1) / 2
        save_image(recon_image.clamp(min=0, max=1)[0], join(output_dir, sample_name, 'recon.png'))


        for transform_idx, transform in enumerate(transforms):

            # get transformation parameters
            translation = torch.tensor(transform['translation'], dtype=torch.float32)
            rot_axis = torch.tensor(transform['rotation_axis'], dtype=torch.float32)
            rot_angle = float(transform['rotation_angle'])

            # transform the foreground object
            edited_img, raw_edited_depth, edited_disparity = diff_handles.transform_foreground(
                depth=depth, prompt=prompt,
                fg_mask=fg_mask, bg_depth=bg_depth,
                inverted_null_text=inverted_null_text, inverted_noise=inverted_noise,
                activations=activations, activations2=activations2, activations3=activations3,
                rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
                use_input_depth_normalization=False)

            # save the edited depth
            save_image((edited_disparity/edited_disparity.max())[0], join(output_dir, sample_name, f'edit_{transform_idx:03d}_disparity.png'))
            save_image((raw_edited_depth/raw_edited_depth.max())[0], join(output_dir, sample_name, f'edit_{transform_idx:03d}_depth_raw.png'))

            # save the edited image
            save_image(edited_img[0], join(output_dir, sample_name, f'edit_{transform_idx:03d}.png'))

def load_diffhandles_inputs(sample_dir, img_res, device):

    # load the input image
    img = load_image(join(sample_dir, 'input.png'))[None, ...]
    if img.shape[-2:] != (img_res, img_res):
        print(f"WARNING: Resizing and cropping image from {img.shape[-2]}x{img.shape[-1]} to {img_res}x{img_res}.")
        img = crop_and_resize(img=img, size=img_res)
    img = img.to(device)

    # load the foreground mask
    fg_mask = load_image(join(sample_dir, 'mask.png'))[None, ...]
    if fg_mask.shape[1] > 1:
        fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
    fg_mask = crop_and_resize(img=fg_mask, size=img_res)
    fg_mask = (fg_mask>0.5).to(device=device, dtype=torch.float32)

    # load the input image depth
    depth = load_depth(join(sample_dir, 'depth.exr'))[None, ...]
    depth = crop_and_resize(img=depth, size=img_res)
    depth = depth.to(device=device, dtype=torch.float32)

    # load the background depth
    bg_depth = load_depth(join(sample_dir, 'bg_depth.exr'))[None, ...]
    bg_depth = crop_and_resize(img=bg_depth, size=img_res)
    bg_depth = bg_depth.to(device=device, dtype=torch.float32)

    return img, fg_mask, depth, bg_depth

if __name__ == '__main__':
    test_diffusion_handles(test_set_path='data/test_set.json', input_dir='data', output_dir='results')
    generate_results_webpage(test_set_path = 'data/test_set.json', website_path = 'results/results.html', relative_image_dir = '.')
