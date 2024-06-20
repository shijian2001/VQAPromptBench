from vqa_datasets import SingleImageQADataset
from vqa_models import ImageQAModel
from prompt_factory import detailed_imageqa_prompt
import numpy as np
from tqdm import tqdm
from typing import *
import json
import os

def build_prompt_func(prompt_template: str):
    def imageqa_prompt(question: str, context: str, choices: List[str]):
        prompt = prompt_template.format(
            question=question,
            context=context,
            choices=choices
        )
        return prompt
    return imageqa_prompt

# load vqa model
# pass default prompt template
vqa_model_name = "deepseek-vl-7b-chat"
vqa_model = ImageQAModel(vqa_model_name, prompt_func=detailed_imageqa_prompt, enable_choice_search=True, enable_interpretation=True, torch_device=0)

# load dataset
benchmark_name = "mmbench"
benchmark = SingleImageQADataset(benchmark_name).get_dataset()

# load prompt templates
prompt_templates = json.load(open("./prompt_factory/prompt_library.json", "r"))["MultiChoiceImageQa"]


logs = {}
logs[vqa_model_name] = {}
logs[vqa_model_name][benchmark_name] = {}
for i, prompt_template in enumerate(prompt_templates):
    print("===============================================================")
    print(f"Evaluated on prompt {i+1}:")
    print("===============================================================")
    logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'] = {}
    logs[vqa_model_name][benchmark_name][f'prompt_{i+1}']["accuracy"] = []
    logs[vqa_model_name][benchmark_name][f'prompt_{i+1}']["explanation"] = []
    for sample in tqdm(benchmark):
        result = vqa_model.multiple_choice_qa(
            data = sample["image"],
            question = sample["question"],
            context=sample["context"],
            choices = sample["choices"],
            answer = sample["answer"],
            prompt_func= build_prompt_func(prompt_template)
        )
        logs[vqa_model_name][benchmark_name][f'prompt_{i+1}']["accuracy"].append(result["accuracy"])
        logs[vqa_model_name][benchmark_name][f'prompt_{i+1}']["explanation"].append(result)
    print(f"Overall Acc for the prompt {i+1}: {np.mean(logs[vqa_model_name][benchmark_name][f'prompt_{i+1}']['accuracy'])}")

# save logs to disk
with open(f'./logs/hallucination_logs/one-stage_{vqa_model_name}_{benchmark_name}_eval.json', "w", encoding='utf-8') as f:
    json.dump(logs, f, ensure_ascii=False, indent=4)

print(f"one stage {vqa_model_name}-{benchmark_name} evaluations have saved successfully!")