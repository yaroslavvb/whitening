#!/bin/sh
# Run python script, filtering out TensorFlow logging
# https://github.com/tensorflow/tensorflow/issues/566#issuecomment-259170351
python $* 3>&1 1>&2 2>&3 3>&- | grep -v ":\ I\ " | grep -v "feature_guard.cc" | grep -v ^pciBusID | grep -v ^major: | grep -v ^name: |grep -v ^Total\ memory:|grep -v ^Free\ memory: | grep -v CUDA_ERROR_NO_DEVICE | grep -v "GCC version"
