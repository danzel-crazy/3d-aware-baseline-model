echo "Running test_diffusion_handles.py"
export cuda_visible_devices=0
DEVICE="cuda:0"
CUDA_VISIBLE_DEVICES=0 python test_diffusion_handles.py \
    --test_set_path /tmp2/danzel/DiffusionHandles/demo_images/test_images/test.json \
    --device $DEVICE \
    --input_dir  /tmp2/danzel/DiffusionHandles/demo_images/test_images \
    --output_dir /tmp2/danzel/DiffusionHandles/test/results/test_images 