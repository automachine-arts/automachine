#!/bin/bash
set -exou pipefail

python3 run_lm_finetuning.v2.1.1.py --no_cuda --output_dir=output --do_train --per_gpu_train_batch_size=1 --train_data_file=usb_key/datasets/tinytinyshakespeare.txt
