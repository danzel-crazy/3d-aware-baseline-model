# export $CUDA_VISIBLE_DEVICES=2

echo "Running seg.py"
# python pipeline.py \
#     --input_folder /tmp2/danzel/zero123/data/test_images  --output_folder /tmp2/danzel/zero123/data/output

# python zero123/generate.py \
#     --input_folder /tmp2/danzel/zero123/results/rotate/benchmark_1/data  --output_folder /tmp2/danzel/zero123/results/rotate/benchmark_1/gen_data
python demo.py --input_dir /tmp2/danzel/zero123/results/rotate/benchmark_1 --output_dir /tmp2/danzel/zero123/results/rotate/benchmark_1/inpaint_results