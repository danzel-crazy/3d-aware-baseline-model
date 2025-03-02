import argparse
from translate_generation import gen_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()
    
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    
    gen_run(args.input_dir, args.output_dir)