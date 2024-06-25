from datasets import load_from_disk, Dataset
from tqdm import tqdm
import json
import copy


def generator():
    dataset = load_from_disk("../subset/llava_instruct_mix_vsft")
    templates  = json.load(open("../prompt_factory/vsft_prompts.json", "r"))["MultiQATemplates"]

    for data in tqdm(dataset):
        messages = data["messages"]
        images = data["images"]

        # Extract the question from user content
        question = next(item['text'] for item in messages[0]['content'] if item['type'] == 'text').strip()

        # Applying all templates
        formatted_questions = [template.format(question=question) for template in templates]

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

gen_dataset = Dataset.from_generator(generator)
gen_dataset = gen_dataset.shuffle(42)
gen_dataset.save_to_disk('../subset/gen_llava_instruct_mix_vsft')
