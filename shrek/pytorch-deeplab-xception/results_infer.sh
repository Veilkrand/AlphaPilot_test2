#!/usr/bin/env bash

# Running Inference
python3 inference.py \
 --checkpoint_path "run/volta-playground/run_04/models/deeplabv3plus-xception_epoch-41.pth" \
 --input_images_path "data/dataset/test/images" \
 --label_images_path "" \
 --result_folder "1024p-os8-epoch41" \
 --imsize 1024 \
