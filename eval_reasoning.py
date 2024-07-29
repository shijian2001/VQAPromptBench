from sentence_transformers import SentenceTransformer, util
from vqa_datasets import SingleImageQADataset
from vqa_models import ImageQAModel
from prompt_factory import detailed_imageqa_prompt
import numpy as np
from tqdm import tqdm
from typing import *
import json

# For all test prompts, we test llava-1.5 (origin/finetuned) on mm-bench (sample 100 data)
# All prompts have 3 elements: question, context, choices
# For each prompt, we load the data into the prompt and add “Please point out the question/context/choices” at the end, and perform three inferences.
# We calculate the ngram of each inferred element and the original element content as a quantitative indicator to measure whether the model can extract the correct information

def calculate_similarity(origin: str, reasoning: str) -> float:
    """Calculate the cosine similarity between two sentences using Sentence Transformers."""
    model = SentenceTransformer("all-mpnet-base-v2", device='cpu')
    
    # Encode the sentences into embeddings
    origin_embedding = model.encode(origin, convert_to_tensor=True)
    reasoning_embedding = model.encode(reasoning, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(origin_embedding, reasoning_embedding).item()
    return similarity


def build_prompt_func(prompt_template: str, reasoning: str):
    def imageqa_prompt(question: str, context: str, choices: List[str]):
        prompt = prompt_template.format(
            question=question,
            context=context,
            choices=" ".join(choices)
        )
        prompt = prompt + f"\nThe query above consists of three elements: question, context, and choices. Please indicate the {reasoning}."
        return prompt 
    return imageqa_prompt

def experiment(
        vqa_model_name: str, 
        benchmark_names: List[str], 
        prompt_templates: List[str]
    ):
    """
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
    vqa_model = ImageQAModel(vqa_model_name, prompt_func=detailed_imageqa_prompt, enable_choice_search=True, torch_device=1, use_lora=False)
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
            
            # batch evaluation
            batch_size = 100
            for i in tqdm(range(0, len(benchmark), batch_size), total=(len(benchmark) + batch_size - 1) // batch_size):
                batch = benchmark[i:i+batch_size]
                for reasoning in ["question", "context", "choices"]:
                    batch_results = vqa_model.batch_verify_reasoning(
                        images = batch["image"],
                        questions = batch["question"],
                        contexts =batch["context"],
                        choices = batch["choices"],
                        answers = batch["answer"],
                        prompt_func= build_prompt_func(prompt_template, reasoning),
                        reasoning = ""
                    )
                batch_accs = [single_results["accuracy"] for single_results in batch_results]
                logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'].extend(batch_accs)

            print(f"Overall Acc for the prompt {i+1}: {np.mean(logs[vqa_model_name][benchmark_name][f'prompt_{i+1}'])}")

    # save logs to disk
    with open(f'./logs/reasoning-finetuning-logs/259k_LLaVaSFTData_30_templates_without_reseaoning_lora_{vqa_model_name}_eval.json', "w", encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    print(f"{vqa_model_name} evaluations have saved successfully!")


experiment(
    vqa_model_name="llavav1.5-7b",
    benchmark_names=["blink", "mmbench", "seedbench1"],
    prompt_templates=json.load(open("./prompt_factory/test_vsft_lora.json", "r"))["MultiChoiceImageQa"]
)