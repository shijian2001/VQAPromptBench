import numpy as np
from tqdm import tqdm
from typing import *
import optuna
import logging
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger("optuna")
logger.setLevel(logging.CRITICAL)


class VQAPromptBayesOptimizer:
    def __init__(
            self, 
            vqa_model, 
            vqa_dataset, 
            prompts_pool: List[str], 
            n_trials: int=10, 
            random_state: int=42, 
            num_samples: Union[None, int]=None, 
            enable_option_ordering: bool=False
        ):

            self.vqa_model = vqa_model

            if num_samples is not None:
                self.vqa_dataset = vqa_dataset.shuffle().select(range(num_samples))
            else:
                self.vqa_dataset = vqa_dataset

            self.prompts_pool = prompts_pool
            self.n_trials = n_trials
            self.random_state = random_state
            self.enable_option_ordering = enable_option_ordering
            self.study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))

            self._construct_loopup_table()

    def _construct_loopup_table(self):
        from sentence_transformers import SentenceTransformer

        # Get embeddings
        print(f"Embedding {len(self.prompts_pool)} prompt templates.....")
        sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        prompt_embeddings = sentence_transformer.encode(self.prompts_pool)
        print("Finish embedding")

        # Construct lookup table
        self.prompt_embedding_tuple_list = [tuple(embedding) for embedding in prompt_embeddings]
        self.prompt_emb_lookup_table = {key: value for key, value in zip(self.prompt_embedding_tuple_list, self.prompts_pool)}

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
            if self.enable_option_ordering:
                result = self.vqa_model.multiple_choice_qa_random_ordering(
                    data=sample["image"],
                    question=sample["question"],
                    context=sample["context"],
                    choices=sample["choices"],
                    answer=sample["answer"],
                    prompt_func=self._build_prompt_func(prompt_template)
                )
            else:
                result = self.vqa_model.multiple_choice_qa(
                    data=sample["image"],
                    question=sample["question"],
                    context=sample["context"],
                    choices=sample["choices"],
                    answer=sample["answer"],
                    prompt_func=self._build_prompt_func(prompt_template)
                )

            accuracy = result["accuracy"]
            accs.append(accuracy)

        metric = np.mean(accs)
        print("Acc:", metric)

        return metric

    def objective(self, trial):
        prompt_emb = trial.suggest_categorical('PromptEmbedding', self.prompt_embedding_tuple_list)
        prompt_template = self.prompt_emb_lookup_table[prompt_emb]
        prompt_index = self.prompts_pool.index(prompt_template) + 1
        print(f"Evaluating prompt_{prompt_index}:\n##############\n{prompt_template}\n##############")
        return self.evaluator(prompt_template)

    def optimize(self):
        self.study.optimize(self.objective, n_trials=self.n_trials)
        best_trial = self.study.best_trial
        best_prompt_emb = best_trial.params['PromptEmbedding']
        best_prompt = self.prompt_emb_lookup_table[best_prompt_emb]
        best_performance = best_trial.value
        return best_prompt, best_performance
