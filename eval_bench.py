from vqa_datasets import SingleImageQADataset
from vqa_models import ImageQAModel
from prompt_factory import detailed_imageqa_prompt
import numpy as np
from tqdm import tqdm
from typing import *
import torch
import json
import os, csv

def build_prompt_func(prompt_template: str):
    def imageqa_prompt(question: str, context: str, choices: List[str]):
        prompt = prompt_template.format(
            question=question,
            context=context,
            choices=" ".join(choices)
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
    # pass use_lora=True to launch [Lora Inference]
    vqa_model = ImageQAModel(vqa_model_name, prompt_func=detailed_imageqa_prompt, enable_choice_search=False, torch_device=1, precision=torch.float16, use_lora=False)
    print("===============================================================")
    print(f"{vqa_model_name} evaluation started:")
    print("===============================================================")

    for benchmark_name in benchmark_names:
        print(f"Evaluated on {benchmark_name}:")
        print("===============================================================")

        logs[vqa_model_name][benchmark_name] = {}
        # load datasets
        benchmark = SingleImageQADataset(benchmark_name).get_dataset()
        statistics = []
        for i, prompt_template in enumerate(prompt_templates):
            print("===============================================================")
            print(f"Evaluated on prompt {i+1}:")
            print("===============================================================")
            logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'] = []

            # single evaluation
            # for sample in tqdm(benchmark):
            #     result = vqa_model.multiple_choice_qa_random_ordering(
            #         data = sample["image"],
            #         question = sample["question"],
            #         context=sample["context"],
            #         choices = sample["choices"],
            #         answer = sample["answer"],
            #         prompt_func= build_prompt_func(prompt_template)
            #     )
            #     logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'].append(result["accuracy"])
            
            # batch evaluation
            batch_size = 100
            for j in tqdm(range(0, len(benchmark), batch_size), total=(len(benchmark) + batch_size - 1) // batch_size):
                batch = benchmark[j:j+batch_size]
                batch_results = vqa_model.batch_qa_extraction(
                    images = batch["image"],
                    questions = batch["question"],
                    contexts = batch["context"],
                    choices = batch["choices"],
                    answers = batch["answer"],
                    prompt_func= build_prompt_func(prompt_template)
                )
                # batch_accs = [single_results["accuracy"] for single_results in batch_results]
                # logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'].extend(batch_accs)
                logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'].extend(batch_results)

            accs, question_extraction_accs, options_extraction_accs = [], [], []
            for single_results in logs[vqa_model_name][benchmark_name][f'prompt_{i+1}']:
                accs.append(single_results["accuracy"])
                question_extraction_accs.append(single_results["question_matched"])
                if single_results["option_matched"] is not None:
                    options_extraction_accs.append(single_results["option_matched"])
            print(f"Overall Acc for the prompt {i+1}: {np.mean(accs)}")
            print(f"Overall question extraction Acc for the prompt {i+1}: {np.mean(question_extraction_accs)}")
            if options_extraction_accs == []:
                options_extraction_accs = [0]
            print(f"Overall option extraction Acc for the prompt {i+1}: {np.mean(options_extraction_accs)}")
        
            statistics.append([i+1, np.mean(accs), np.mean(question_extraction_accs), np.mean(options_extraction_accs)])
        
        with open(f'./logs/reasoning-finetuning-logs/{vqa_model_name}_statistics_{benchmark_name}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Prompt', 'Overall Acc', 'Question Extraction Acc', 'Option Extraction Acc'])
            writer.writerows(statistics)

    # save logs to disk
    # './logs/reasoning-finetuning-logs/259k_LLaVaSFTData_30_templates_without_reseaoning_{vqa_model_name}_eval.json'
    with open(f'./logs/reasoning-finetuning-logs/7_29_259k_LLaVaSFTData_30_templates_without_reseaoning_{vqa_model_name}_eval.json', "w", encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    print(f"{vqa_model_name} evaluations have saved successfully!")


experiment(
    vqa_model_name="llavav1.5-7b-finetuned",
    benchmark_names=["seedbench1"],
    prompt_templates=json.load(open("./prompt_factory/test_vsft_lora.json", "r"))["MultiChoiceImageQa"]
)