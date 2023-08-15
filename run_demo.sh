#!/bin/bash

# Specify the paths and parameters
image_folder="/home/aymen/PycharmProjects/owlvit_segment_anything/detection_dataset/images/val/"  # Replace with your image folder path
text_prompt="field,road"
annotation_output="/home/aymen/PycharmProjects/owlvit_segment_anything/detection_dataset/labels/val/"  # Replace with your output folder path
device="cuda:0"
get_topk="--get_topk"
box_threshold="0.1"
output_dir="outputs"

# Loop through image files in the folder
for image_file in "$image_folder"/*; do
    if [[ -f "$image_file" ]]; then
        python demo.py --image_path "$image_file" --text_prompt "$text_prompt" \
                       --annotation_output "$annotation_output" --device "$device" \
                       $get_topk --box_threshold "$box_threshold" --output_dir "$output_dir"
    fi
done
