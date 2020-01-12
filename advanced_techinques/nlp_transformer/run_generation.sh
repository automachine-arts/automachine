#!/bin/bash
set -exou pipefail

# python ./pytorch_run_generation.py --model_type=gpt2 --length=20 --model_name_or_path=gpt2

python ./pytorch_run_generation.py \
    --model_type=gpt2 \
    --length=20 \
    --model_name_or_path=gpt2 \

