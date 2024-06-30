import numpy as np
from tqdm import tqdm
from typing import *
from skopt import gp_minimize
from skopt.space import Categorical

class VQAPromptBayesOptimizer:
    def __init__(self, vqa_model, vqa_dataset, prompts_pool, n_calls=10, random_state=42):
        self.vqa_model = vqa_model
        self.vqa_dataset = vqa_dataset
        self.prompts_pool = prompts_pool
        self.n_calls = n_calls
        self.random_state = random_state

        self.search_space = [Categorical(prompts_pool, name='PromptTemplate')]

    def _build_prompt_func(self, prompt_template: str):
        def imageqa_prompt(question: str, context: str, choices: List[str]):
            prompt = prompt_template.format(
                question=question,
                context=context,
                choices=choices
            )
            return prompt
        return imageqa_prompt

    def evaluator(self, prompt_template):
        accs = []
        for sample in tqdm(self.vqa_dataset):
            result = self.vqa_model.multiple_choice_qa(
                data = sample["image"],
                question = sample["question"],
                context=sample["context"],
                choices = sample["choices"],
                answer = sample["answer"],
                prompt_func= self._build_prompt_func(prompt_template)
            )
            accuracy = result["accuracy"]
            accs.append(accuracy)
        return np.mean(accs)

    def objective(self, params):
        prompt_template = params[0]
        return -self.evaluator(prompt_template)

    def optimize(self):
        res = gp_minimize(self.objective, self.search_space, n_calls=self.n_calls, random_state=self.random_state)
        best_prompt = res.x[0]
        best_performance = -res.fun
        return best_prompt, best_performance