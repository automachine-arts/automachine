#!/bin/bash
set -exou pipefail

conda init $(echo $SHELL | sed 's:.*/::g')
conda activate tf
