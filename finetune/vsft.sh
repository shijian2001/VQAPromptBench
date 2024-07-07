#!/bin/bash

scripts=("vsft_llava_next.py" "vsft_idefics.py" "vsft_phi3vision.py")

data_hub_pairs=(
    "shijianS01/20-templates-llava-vsft-300k 20-templates"
    "shijianS01/6k-templates-mm-vsft-300k 6k-templates"
)

for script in "${scripts[@]}"; do
    for pair in "${data_hub_pairs[@]}"; do
        set -- $pair
        data_path=$1
        hub_model_id=$2

        torchrun --standalone --nnodes=1 --nproc_per_node=8 $script --data_path $data_path --hub_model_id $hub_model_id --output_dir "./outputs"
    done
done
