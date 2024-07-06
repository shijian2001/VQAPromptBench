## 🌟 Install
```
pip install -U pip
pip install -r requirements.txt
```
Remember to check if *deepspeed* is installed successfully

Then you can download the three finetune scipts and *zero_stage3_config.json* directly.

Please put *zero_stage3_config.json* and finetune scripts in the same directory

## 🌟 Run

```
# if you have 8 GPUs
# please modify the output_dir
torchrun --standalone --nnodes=1 --nproc_per_node=8 vsft_phi3vision.py --output_dir "./output"
```