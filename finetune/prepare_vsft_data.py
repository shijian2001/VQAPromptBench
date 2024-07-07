# Visual instruction tuning data generator
# DataRichPromptLessGenerator:
    # Data: 15k samples from HuggingFaceH4/llava-instruct-mix-vsft
    # Prompt Templates: 20 templates
    # shijianS01/20-templates-llava-vsft-300k
# PromptRichDataLessGenerator:
    # Data: 50 samples from MMBench
    # Prompt Templates: 6000 templates
    # shijianS01/6k-templates-mm-vsft-300k

from typing import List
from datasets import load_from_disk, load_dataset, Dataset
from tqdm import tqdm
from typing import *
import json
import copy

class BaseGenerator():
    def __init__(self, dataset_path: str, templates: List[str]):
        # self.dataset = load_dataset(dataset_path)
        self.dataset = load_from_disk(dataset_path)
        self.templates = templates
    
    def generator(self):
        "(Abstract method) abstract data generator method"

class DataRichPromptLessGenerator(BaseGenerator):
    def __init__(self, dataset_path: str, templates: List[str]):
        super().__init__(dataset_path, templates)

    def generator(self):
        for data in tqdm(self.dataset):
            messages = data["messages"]
            images = data["images"]

            # Extract the question from user content
            question = next(item['text'] for item in messages[0]['content'] if item['type'] == 'text').strip()

            # Applying all templates
            formatted_questions = [template.format(question=question) for template in self.templates]

            # Generate new samples
            gen_messages = []
            for formatted_question in formatted_questions:
                gen_message = copy.deepcopy(messages)
                for item in gen_message[0]["content"]:
                    if item["type"] == "text":
                        item["text"] = formatted_question
                gen_messages.append(gen_message)

            all_gen_messages = []
            for item in gen_messages:
                gen_item_1, gen_item_2 = copy.deepcopy(item), copy.deepcopy(item)

                if item[0]["content"][0]["type"] == "image":
                    gen_item_1[0]["content"][1]["text"] = "\n" + gen_item_1[0]["content"][1]["text"]
                    gen_item_2[0]["content"][0], gen_item_2[0]["content"][1] = gen_item_2[0]["content"][1], gen_item_2[0]["content"][0]
                    gen_item_2[0]["content"][0]["text"] = gen_item_2[0]["content"][0]["text"] + "\n"
                else:
                    gen_item_1[0]["content"][0]["text"] = gen_item_1[0]["content"][0]["text"] + "\n"
                    gen_item_2[0]["content"][0], gen_item_2[0]["content"][1] = gen_item_2[0]["content"][1], gen_item_2[0]["content"][0]
                    gen_item_2[0]["content"][1]["text"] = "\n" + gen_item_2[0]["content"][1]["text"]
                all_gen_messages.append(gen_item_1)
                all_gen_messages.append(gen_item_2)    

            all_gen_samples = [{"messages": message, "images": images} for message in all_gen_messages]

            for sample in all_gen_samples:
                yield sample

class PromptRichDataLessGenerator(BaseGenerator):
    def __init__(self, dataset_path: str, templates: List[str]):
        super().__init__(dataset_path, templates)

    def _make_options(self, choices, format='letter'):
        assert format in ['numeric', 'letter']
        if format == 'numeric':
            prefix1 = [str(i + 1) for i in range(len(choices))]
        else:
            prefix1 = [chr(ord("a") + i).upper() for i in range(len(choices))]
        prefix2 = [f"({p})" for p in prefix1]
        return prefix1, prefix2, [f'{p} {c}' for p, c in zip(prefix2, choices)]

    def generator(self):
        for data in tqdm(self.dataset):
            prompts = [
                template.format(
                    question = data["question"],
                    context = data["context"],
                    choices = self._make_options(data["choices"])[2]
                ) for template in self.templates
            ]

            for prompt in prompts:
                vsft_data = {
                    'messages': [
                        {
                            'content': [
                                {
                                    'index': 0, 
                                    'text': None, 
                                    'type': 'image'
                                },
                                {
                                    'index': None,
                                    'text': prompt,
                                    'type': 'text'
                                }
                            ],
                            'role': 'user'
                        },
                        {
                            'content': [
                                {
                                    'index': None,
                                    'text': data["answer"],
                                    'type': 'text'
                                }
                            ],
                            'role': 'assistant'
                        }
                    ],
                    'images': data["image"]
                }

                yield vsft_data

if __name__ == "__main__":
    generator = PromptRichDataLessGenerator(
        dataset_path="../subset/meta_mm_vsft",
        templates=json.load(open("../prompt_factory/prompt_pool.json"))["MultiChoiceImageQa"]
    ).generator
    gen_dataset = Dataset.from_generator(generator)
    gen_dataset = gen_dataset.shuffle(42)
    # gen_dataset.save_to_disk('../subset/gen_llava_instruct_mix_vsft')
    print(len(gen_dataset))
    dataset = gen_dataset.train_test_split(test_size=0.1)
    print(dataset)
    # dataset.push_to_hub("shijianS01/6k-templates-mm-vsft-300k")
