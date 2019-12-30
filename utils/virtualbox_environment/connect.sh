#!/bin/bash
set -uxo pipefail

ssh mlart@localhost:2222
while [ $? -ne 0 ]; do
  sleep 1
  ssh mlart@localhost:2222
done
