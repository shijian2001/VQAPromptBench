## 🌟 Install
```
pip install -U pip
pip install -r requirements.txt
```
Remember to check if ```deepspeed``` is installed successfully

Then you can download the three finetune scripts, ```vsft.sh``` and ```zero_stage3_config.json``` directly.

Please put ```zero_stage3_config.json``` and these scripts in the same directory

## 🌟 Run
```
bash vsft.sh
```
You can also run the following command separately
```
# If you have 8 GPUs
# You can modify the output_dir
torchrun --standalone --nnodes=1 --nproc_per_node=8 vsft_llava_next.py --data_path "shijianS01/6k-templates-mm-vsft-300k" --hub_model_id "6k-templates" --output_dir "./output"
```