from vqa_datasets import SingleImageQADataset
from vqa_models import ImageQAModel
from prompt_factory import detailed_imageqa_prompt
from prompt_optimization import VQAPromptBayesOptimizer
import json

# prompts search space
prompts_pool = json.load(open("prompt_factory/prompt_library.json", "r"))["MultiChoiceImageQa"]

# load vqa model
vqa_model = ImageQAModel("deepseek-vl-7b-chat", prompt_func=detailed_imageqa_prompt, enable_choice_search=True, torch_device=1)

# load vqa dataset
vqa_dataset = SingleImageQADataset("mmbench").get_dataset()

# define bayes searcher
optimizer = VQAPromptBayesOptimizer(vqa_model, vqa_dataset, prompts_pool, n_trials=10, random_state=42)
best_prompt, best_performance = optimizer.optimize()
best_prompt_index = prompts_pool.index(best_prompt) + 1
print(f"Best prompt is prompt_{best_prompt_index}:\n##############\n{best_prompt}\n##############")
print(f"Best Acc: {best_performance:.4f}")