#!/bin/bash

python run_generation.py \
    --task insert \
    --checkpoint_path /tmp2/danzel/object-edit/checkpoints/rotate.ckpt \
    --image_path demo_images/rotate_mug.jpg \
    --object_prompt "white mug" \
    --rotation_angle 90 \
    --device 1 \
    --cfg_scale 3.0