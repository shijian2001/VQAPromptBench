from singe_imageqa_datasets import SingleImageQADataset
from imageqa_model import ImageQAModel
from prompt import detailed_imageqa_prompt
from tqdm import tqdm
from typing import *
import os
os.environ['HF_HOME'] = '/linxindisk/.cache/huggingface/'

### TODO
# -1. prompts: prompt_library.json
#     -- 50 prompts
# -2. class for vqa models: imageqa_model.py
#     -- DeepSeek


def experiment(
        vqa_model_name: str, 
        benchmark_names: List[str], 
        prompts: List[str],
        logs
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

    for benchmark_name in benchmark_names:
        logs[vqa_model_name][benchmark_name] = {}
        # load datasets
        benchmark = SingleImageQADataset(benchmark_name).get_dataset()
        # for dataset in datasets:
        for i, prompt in enumerate(prompts):
            logs[vqa_model_name][benchmark_name][f'prompt_{i}'] = []
            for sample in tqdm(benchmark):
                result = vqa_model.multiple_choice_qa_random_ordering(
                    data = sample["image"],
                    question = sample["question"],
                    choices = sample["choices"],
                    answer = sample["answer"],
                    # prompt_func= 
                )
                logs[vqa_model_name][benchmark_name][f'prompt_{i}'].append(result["accuracy"])
                # print(result["accuracy"])

    # save logs to disk