import argparse
import torch
# from maske_extracter.seg import seg
from mask_extracter.seg import seg
from test_diffusion_handles import test_diffusion_handles

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set_path', type=str, default='data/photogen/photogen.json')
    parser.add_argument('--input_dir', type=str, default='data/photogen')
    parser.add_argument('--output_dir', type=str, default='results/photogen')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--cache_input_image_identity', action='store_true')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f"Using device: {args.device}")
    if args.device is not None:
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    seg(input_dir=args.input_dir, test_set_path=args.test_set_path)
    
    test_diffusion_handles(
        test_set_path=args.test_set_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        cache_input_image_identity=args.cache_input_image_identity,
        config_path=args.config_path,
        device=args.device
    )
        