from vqa_datasets import SingleImageQADataset
from vqa_models import ImageQAModel
from prompt_factory import detailed_imageqa_prompt
from prompt_optimization import VQAPromptBayesOptimizer
import json

# load vqa model
vqa_model = ImageQAModel("deepseek-vl-7b-chat", prompt_func=detailed_imageqa_prompt, enable_choice_search=True, torch_device=0)

# load vqa dataset
vqa_dataset = SingleImageQADataset("mmbench").get_dataset()

# prompts search space
# TODO: Add prompt generator in a programmatic way
prompts_pool = json.load(open("prompt_factory/prompt_library.json", "r"))["MultiChoiceImageQa"]

# define bayes searcher
optimizer = VQAPromptBayesOptimizer(vqa_model, vqa_dataset, prompts_pool, n_calls=10, random_state=42)
best_prompt, best_performance = optimizer.optimize()
print(f"Best prompt: {best_prompt}")
print(f"Best Acc: {best_performance:.4f}")