
# Main method: bg inpainting only

cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/../data/rotate/blue_fucet outdir=$(pwd)/../data/inpaint_results
cd ..

# Optional: Anydoor object inpainting
# cd Anydoor
# python run_inference.py
# cd ..

# Optional: paint-by-example object inpainting
# cd Paint-by-Example
# python scripts/inference.py \
# --plms --outdir results \
# --config configs/v1.yaml \
# --ckpt checkpoints/model.ckpt \
# --image_path examples/image/example_1.png \
# --mask_path examples/mask/example_1.png \
# --reference_path examples/reference/example_1.jpg \
# --seed 321 \
# --scale 5
# cd ..