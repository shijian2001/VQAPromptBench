## 🌟 Install
```
pip install -U pip
pip install -r requirements.txt
```
Then you can download the three finetune scipts directly

## 🌟 Run
For each script, you should modify the **output_dir** in training_args.

The you can run like following:
```
# if you have 8 GPUs
accelerate launch vsft_phi3vision.py --num_processes=8
```