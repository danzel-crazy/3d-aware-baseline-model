echo "Running pipeline.py"
export cuda_visible_devices=0
DEVICE="cuda:0"
CUDA_VISIBLE_DEVICES=0 python pipeline.py \
    --test_set_path /tmp2/danzel/DiffusionHandles/pipeline/images/test.json \
    --device $DEVICE \
    --input_dir  /tmp2/danzel/DiffusionHandles/pipeline/images \
    --output_dir /tmp2/danzel/DiffusionHandles/pipeline/output 