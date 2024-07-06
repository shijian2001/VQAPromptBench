# Visual instruction tuning data generator
# DataRichPromptLessGenerator:
    # Data: 15k samples from HuggingFaceH4/llava-instruct-mix-vsft
    # Prompt Templates: 20 templates
# PromptRichDataLessGenerator:
    # Data: 50 samples from MMBench
    # Prompt Templates: 6000 templates

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
        pass

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

    def generator(self):
        pass

if __name__ == "__main__":
    generator = DataRichPromptLessGenerator(
        dataset_path="../subset/llava_instruct_mix_vsft",
        templates=json.load(open("../prompt_factory/vsft_prompts.json", "r"))["MultiQATemplates"]
    ).generator
    gen_dataset = Dataset.from_generator(generator)
    gen_dataset = gen_dataset.shuffle(42)
    # gen_dataset.save_to_disk('../subset/gen_llava_instruct_mix_vsft')
    print(len(gen_dataset))
