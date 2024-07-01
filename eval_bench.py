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
    vqa_model = ImageQAModel(vqa_model_name, prompt_func=detailed_imageqa_prompt, enable_choice_search=True, torch_device=1)
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
            print(f"Evaluated on prompt {i+1}:")
            print("===============================================================")
            logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'] = []
            for sample in tqdm(benchmark):
                result = vqa_model.multiple_choice_qa_random_ordering(
                    data = sample["image"],
                    question = sample["question"],
                    context=sample["context"],
                    choices = sample["choices"],
                    answer = sample["answer"],
                    prompt_func= build_prompt_func(prompt_template)
                )
                logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'].append(result["accuracy"])
            print(f"Overall Acc for the prompt {i+1}: {np.mean(logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'])}")

    # save logs to disk
    with open(f'./logs/eval_logs/{vqa_model_name}_eval.json', "w", encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    print(f"{vqa_model_name} evaluations have saved successfully!")


experiment(
    vqa_model_name="phi-3-vision",
    benchmark_names=["blink", "mmbench", "seedbench1"],
    prompt_templates=json.load(open("./prompt_factory/prompt_library.json", "r"))["MultiChoiceImageQa"]
)