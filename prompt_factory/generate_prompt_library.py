from itertools import permutations
import json
from typing import *

base_vqa_prompt_elems = [
    ["{context}\n", "{question}\n", "{choices}\n"],
    ["Context: {context}\n", "Question: {question}\n", "Choices: {choices}\n"],
    ["According to the context\nContext: {context}\n", "Question: {question}\n", "Choices: {choices}\n"],
    ["Context: {context}\n", "Based on the image, answer the following question\nQuestion: {question}\n", "Choices: {choices}\n"],
    ["Context: {context}\n", "Question: {question}\n", "Select from the following choices\nChoices: {choices}\n"],
    ["According to the context\nContext: {context}\n", "Based on the image, answer the following question\nQuestion: {question}\n", "Choices: {choices}\n"],
    ["According to the context\nContext: {context}\n", "Question: {question}\n", "Select from the following choices\nChoices: {choices}\n"],
    ["Context: {context}\n", "Based on the image, answer the following question\nQuestion: {question}\n", "Select from the following choices\nChoices: {choices}\n"],
    ["According to the context\nContext: {context}\n", "Based on the image, answer the following question\nQuestion: {question}\n", "Select from the following choices\nChoices: {choices}\n"]
]

def generate_prompt_templates(elements: List[str]):
    perm = permutations(elements)
    prompt_templates = [''.join(p).strip() for p in perm]
    return prompt_templates

if __name__ == "__main__":
    prompt_templates = []
    for elems in base_vqa_prompt_elems:
        prompt_templates.extend(generate_prompt_templates(elems))

    # generate prompt library
    prompt_library = {}
    prompt_library["DirectImageQa"]=[]
    prompt_library["MultiChoiceImageQa"]=prompt_templates
    with open('./prompt_library.json', "w", encoding='utf-8') as f:
        json.dump(prompt_library, f, ensure_ascii=False, indent=4)
    
    # for prompt_template in prompt_templates:
    #     print(prompt_template)
    #     print("====================================")