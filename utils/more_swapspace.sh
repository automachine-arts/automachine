#!/bin/bash
set -exou pipefail

sudo fallocate -l 4G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
