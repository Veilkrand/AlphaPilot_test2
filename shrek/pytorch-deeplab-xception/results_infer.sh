#!/usr/bin/env bash

# Running Inference
python3 inference.py \
 --checkpoint_path "run/run_03/models/deeplabv3plus-xception_epoch-9.pth" \
 --input_images_path "data/dataset/test/images" \
 --label_images_path "" \
 --result_folder "512pos8" \
 --imsize 512 \
