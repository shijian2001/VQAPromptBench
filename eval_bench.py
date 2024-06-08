from singe_imageqa_datasets import SingleImageQADataset
from imageqa_model import ImageQAModel
from prompt import detailed_imageqa_prompt
import numpy as np
from tqdm import tqdm
from typing import *
import json
import os
os.environ['HF_HOME'] = '/linxindisk/.cache/huggingface/'

### TODO
# -1. prompts: prompt_library.json
#     -- 50 prompts
# -2. class for vqa models: imageqa_model.py
#     -- DeepSeek

def build_prompt_func(prompt_template: str):
    def imageqa_prompt(question, choices: List[str]):
        prompt = prompt_template.format(question, choices)
        return prompt
    return imageqa_prompt

def experiment(
        vqa_model_name: str, 
        benchmark_names: List[str], 
        prompt_templates: List[str]
    ):
    """
    - Evaluate just one VQA model on all benchmark datasets with all prompts
    - Log and saved evaluation results
        - results: VQA results dict
        - Saved: 
        {
            vqa_model:{
                benchmark:{
                    prompt: [vqa acc for every data]
                }
            }
        }
    """

    logs = {}
    logs[vqa_model_name] = {}

    # load vqa model
    # pass default prompt template
    vqa_model = ImageQAModel(vqa_model_name, prompt_func=detailed_imageqa_prompt, enable_choice_search=True)
    print("===============================================================")
    print(f"{vqa_model_name} evaluation started:")
    print("===============================================================")

    for benchmark_name in benchmark_names:
        print(f"Evaluated on {benchmark_name}:")
        print("===============================================================")

        logs[vqa_model_name][benchmark_name] = {}
        # load datasets
        benchmark = SingleImageQADataset(benchmark_name).get_dataset()

        for i, prompt_template in enumerate(prompt_templates):
            print("===============================================================")
            print(f"Evaluated on prompt {i}:")
            print("===============================================================")
            logs[vqa_model_name][benchmark_name][f'prompt_{i}'] = []
            for sample in tqdm(benchmark):
                result = vqa_model.multiple_choice_qa_random_ordering(
                    data = sample["image"],
                    question = sample["question"],
                    choices = sample["choices"],
                    answer = sample["answer"],
                    prompt_func= build_prompt_func(prompt_template)
                )
                logs[vqa_model_name][benchmark_name][f'prompt_{i}'].append(result["accuracy"])
            print(f"Overall Acc for the prompt {i}: {np.mean(logs[vqa_model_name][benchmark_name][f'prompt_{i}'])}")

    # save logs to disk
    with open(f'./logs/{vqa_model_name}_eval.json', "w", encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    print(f"{vqa_model_name} evaluations have saved successfully!")


experiment(
    vqa_model_name="qwenvl-chat",
    benchmark_names=["blink"],
    prompt_templates=json.load(open("./prompt_library.json", "r"))["multi-choices image qa"]
)