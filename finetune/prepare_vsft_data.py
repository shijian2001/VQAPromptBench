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
from datasets import DatasetDict
from tqdm import tqdm
from typing import *
import json
import copy
import random

class BaseGenerator():
    def __init__(self, dataset_path: str, templates: List[str], split="None"):
        self.templates = templates

        # special case
        original_templates = ["{question}" for _ in range(len(self.templates)-2)]
        self.templates = original_templates + self.templates
        # print(self.templates)

        if split != "None":
            self.dataset = load_dataset(dataset_path, split=split)
        else:
            self.dataset = load_dataset(dataset_path)
    
    def generator(self):
        "(Abstract method) abstract data generator method"

class DataRichPromptRandomGenerator(BaseGenerator):
    def __init__(self, dataset_path: str, templates: List[str], split="None"):
        super().__init__(dataset_path, templates, split)

    def generator(self):
        for data in tqdm(self.dataset):
            vsft_data = copy.deepcopy(data)

            vsft_data["images"] = data["images"][0] # important for calling the image decoding feature

            messages = vsft_data["messages"]

            for conversation in messages:
                if conversation["role"] == "user":
                    origin_content = conversation["content"]
                    for item in origin_content:
                        if item["type"] == "text":
                            template = random.choice(self.templates)
                            item["text"] = template.format(question=item["text"])
            
            yield vsft_data


class DataRichPromptRandomReasoningGenerator(BaseGenerator):
    def __init__(self, dataset_path: str, templates: List[str], split="None"):
        super().__init__(dataset_path, templates, split)

    def generator(self):
        for data in tqdm(self.dataset):
            vsft_data = copy.deepcopy(data)

            vsft_data["images"] = data["images"][0]

            messages = vsft_data["messages"]

            assert len(messages) % 2 == 0, "Messages length is not even"

            for i in range(0, len(messages), 2):
                question, answer = tuple(messages[i:i + 2])
                
                # question
                origin_question_content = question["content"]
                for item in origin_question_content:
                    if item["type"] == "text":
                        origin_q = copy.deepcopy(item["text"])
                        if origin_q[0] == "\n": 
                            origin_q = origin_q[1:]
                        template = random.choice(self.templates)
                        item["text"] = template.format(question=item["text"])
                
                # answer
                origin_answer_content = answer["content"]
                assert len(origin_answer_content) == 1, "Only one answer"
                origin_a = copy.deepcopy(origin_answer_content[0]["text"])
                origin_answer_content[0]["text"] = f"Question: {origin_q}\n{origin_a}"
            
            yield vsft_data

    
class DataRichPromptLessGenerator(BaseGenerator):
    def __init__(self, dataset_path: str, templates: List[str], split="None"):
        super().__init__(dataset_path, templates, split)

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
    print("Train Generator")
    train_generator = DataRichPromptRandomReasoningGenerator(
        dataset_path="HuggingFaceH4/llava-instruct-mix-vsft",
        templates=json.load(open("../prompt_factory/vsft_prompts.json"))["MultiQATemplates"],
        split = "train"
    ).generator

    print("Test Generator")
    test_generator = DataRichPromptRandomReasoningGenerator(
        dataset_path="HuggingFaceH4/llava-instruct-mix-vsft",
        templates=json.load(open("../prompt_factory/vsft_prompts.json"))["MultiQATemplates"],
        split = "test"
    ).generator

    test_dataset = Dataset.from_generator(test_generator)
    print(test_dataset[0]["images"])
    print(test_dataset[0]["messages"][1])

    train_dataset = Dataset.from_generator(train_generator)

    # Combine into a DatasetDict
    gen_dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # gen_dataset = gen_dataset.shuffle(42)
    # gen_dataset.save_to_disk('../subset/random_prompts_llava__vsft')
    # dataset = gen_dataset.train_test_split(test_size=0.1)
    print(gen_dataset)
    gen_dataset.push_to_hub("shijianS01/reasoning-30-templates-259k-llava-data")
