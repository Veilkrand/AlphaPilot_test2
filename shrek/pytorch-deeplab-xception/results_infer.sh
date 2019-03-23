#!/usr/bin/env bash

# Running Inference
python3 inference.py \
 --checkpoint_path "run/run_05/models/deeplabv3plus-xception_epoch-53.pth" \
 --input_images_path "data/dataset/test/images" \
 --label_images_path "data/dataset/test/labels" \
 --result_folder "512p-os8-aug-affine-run05-epoch53-test" \
 --imsize 512 \
 --conf_threshold 0.5
