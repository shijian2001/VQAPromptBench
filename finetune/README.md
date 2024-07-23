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
# mix-random
torchrun --standalone --nnodes=1 --nproc_per_node=1 vsft_llava.py --data_path "shijianS01/mix-random-templates-llava-vsft-259k" --hub_model_id "mix-random-templates" --output_dir "../logs/test_output/test_mix" --use_lora=True --use_4_bit=True

# random
torchrun --standalone --nnodes=1 --nproc_per_node=1 vsft_llava.py --data_path "shijianS01/random-templates-llava-vsft-259k" --hub_model_id "random-templates" --output_dir "../logs/test_output/test_random" --use_lora=True --use_4_bit=True
```