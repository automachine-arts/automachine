#!/bin/bash

# WE WILL BE USING PRE-TRAINED INCEPTION V3 WEIGHTS GIVEN FROM USBs, SINCE
# THEY ARE TOO LARGE TO DOWNLOAD. THE FILENAME ON THE USB IS
# inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5.
#
# Unix users: move inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 to
# $HOME/.keras/models
#
# Windows users: move inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
# to %USERPROFILE%\.keras\models

set -exou pipefail

conda install pillow