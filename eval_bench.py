from vqa_datasets import SingleImageQADataset
from transformers import set_seed
from vqa_models import ImageQAModel
from prompt_factory import detailed_imageqa_prompt
import numpy as np
from tqdm import tqdm
from typing import *
import torch
import json
import argparse
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
        prompt_templates: List[str],
        seed: int=42
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

    set_seed(seed=seed)

    logs = {}
    logs[vqa_model_name] = {}

    # load vqa model
    # pass default prompt template
    # pass use_lora=True to launch [Lora Inference]
    vqa_model = ImageQAModel(vqa_model_name, prompt_func=detailed_imageqa_prompt, enable_choice_search=True, torch_device=1, precision=torch.float16, use_lora=False)
    print("===============================================================")
    print(f"{vqa_model_name} evaluation started:")
    print("===============================================================")

    for benchmark_name in benchmark_names:
        print(f"Evaluated on {benchmark_name}:")
        print("===============================================================")

        logs[vqa_model_name][benchmark_name] = {}
        # load datasets
        benchmark = SingleImageQADataset(benchmark_name).get_dataset()
        # benchmark = benchmark.select([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        print(f"Benchmark Length: {len(benchmark)}")
        # statistics = []
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
            batch_size = 20
            for j in tqdm(range(0, len(benchmark), batch_size), total=(len(benchmark) + batch_size - 1) // batch_size):
                batch = benchmark[j:j+batch_size]
                batch_results = vqa_model.batch_multiple_choice_qa_random_ordering(
                    images = batch["image"],
                    questions = batch["question"],
                    contexts = batch["context"],
                    choices = batch["choices"],
                    answers = batch["answer"],
                    prompt_func= build_prompt_func(prompt_template)
                )
                batch_accs = [single_results["accuracy"] for single_results in batch_results]
                logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'].extend(batch_accs)
                # logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'].extend(batch_results)

            # accs, question_extraction_accs, options_extraction_accs = [], [], []
            # for single_results in logs[vqa_model_name][benchmark_name][f'prompt_{i+1}']:
            #     accs.append(single_results["accuracy"])
            #     question_extraction_accs.append(single_results["question_matched"])
            #     if single_results["option_matched"] is not None:
            #         options_extraction_accs.append(single_results["option_matched"])
            print(f"Overall Acc for the prompt {i+1}: {np.mean(logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'])}")
            # print(f"Overall question extraction Acc for the prompt {i+1}: {np.mean(question_extraction_accs)}")
            # if options_extraction_accs == []:
            #     options_extraction_accs = [0]
            # print(f"Overall option extraction Acc for the prompt {i+1}: {np.mean(options_extraction_accs)}")
        
            # statistics.append([i+1, np.mean(accs), np.mean(question_extraction_accs), np.mean(options_extraction_accs)])
        
        # with open(f'./logs/reasoning-finetuning-logs/{vqa_model_name}_statistics_{benchmark_name}.csv', mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Prompt', 'Overall Acc', 'Question Extraction Acc', 'Option Extraction Acc'])
        #     writer.writerows(statistics)

    # save logs to disk
    # ./logs/multi-templates-logs/100_samples_best_3_epoch_mask_259k_llava_data_generator_templates_{vqa_model_name}_eval.json
    # ./logs/multi-templates-logs/100_samples_{vqa_model_name}_eval.json
    with open(f'./logs/template-generator/evaluation/mmbench/100_samples_{vqa_model_name}_eval.json', "w", encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    print(f"{vqa_model_name} evaluations have saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa_model", type=str, required=True, help="VQA Model Name")

    args = parser.parse_args()

    experiment(
        vqa_model_name=args.vqa_model,
        benchmark_names=["mmbench"],
        prompt_templates=json.load(open("./prompt_factory/held_out_prompts.json", "r"))["MultiChoiceImageQa"],
        seed=42
    )