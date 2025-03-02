#!/bin/bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=1
# Run the test.py script

python test_trans.py --input_dir /tmp2/danzel/3d-aware-baseline-model/object3dit_translate/unseen --output_dir /tmp2/danzel/3d-aware-baseline-model/object3dit_translate/output