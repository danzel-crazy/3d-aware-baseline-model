#!/bin/bash

# Define the base directory containing the images
BASE_DIR="test_images"

# Define the save directory
SAVE_DIR="test_outputs/rotate_new"

# Loop through all input.jpg files in subdirectories
find "$BASE_DIR" -type f \( -name "input.jpg" -o -name "input.png" \) | while read -r image_path; do
# find "$BASE_DIR" -type f -name "input.jpg" | while read -r image_path; do
    # Extract the parent folder name as object prompt
    object_prompt=$(basename "$(dirname "$image_path")")
    
    # Extract the relative path for saving
    relative_path="${image_path#$BASE_DIR/}"
    relative_dir=$(dirname "$relative_path")

    # Run the Python script for each rotation angle
    for angle in 90 180 270 360; do
        echo "Processing: Object Prompt: '$object_prompt', Rotation Angle: $angle"

        python run_generation.py \
            --task rotate \
            --checkpoint_path /tmp2/danzel/object-edit/checkpoints/rotate.ckpt \
            --image_path "$image_path" \
            --object_prompt "$object_prompt" \
            --rotation_angle "$angle" \
            --device 1 \
            --cfg_scale 3.0 \
            --save_dir "$SAVE_DIR/$relative_dir/angle_$angle"
    done
done
