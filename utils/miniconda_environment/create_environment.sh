#!/bin/bash
set -exou pipefail

[ -f /opt/miniconda3/etc/profile.d/conda.sh ] && source /opt/miniconda3/etc/profile.d/conda.sh

conda create -n automachine tensorflow pytorch
