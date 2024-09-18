import itertools
import json

questions = [
    "Please answer the question which is about the image:\n{question}",
    "You should answer this question:\n{question}",
    "Taking the image into account, answer the question: {question}",
    "Read the question which is about the picture: {question}",
    "Read the question and give your answer: {question}",
]

contexts = [
    "Consider the context of the image when answering: {context}",
    "The context you need to consider regarding the image is: {context}",
    "To answer the question, you should read this context:\n{context}",
    "Please refer to the context below for information about the image: {context}",
    "The context below is relevant to this question: {context}"
]

choices = [
    "You should choose the best answer from these choices:\n{choices}",
    "The best answer is in these choices: {choices}",
    "The best answer to the question is included in these choices:\n{choices}",
    "Please give a right answer to the question from these choices: {choices}",
    "Consider the following options related to the question:\n{choices}",
    "The correct answer can be selected from these options:\n{choices}",
    "The right answer to the question can be chosen from these options:\n{choices}",
    "To find the correct answer, you need to consider the following options: {choices}",
    "Among these options, you can find the correct answer: {choices}",
    "The right choice for the question can be found among these options:\n{choices}",
    "To find the best answer, you can choose from these options:\n{choices}"
]

elements = [questions, contexts, choices]

vqa_prompt_pool = []
for ordering in itertools.permutations(elements):
    for combination in itertools.product(*ordering):
        prompt = "\n".join(combination).strip()
        vqa_prompt_pool.append(prompt)

prompt_pool = {}
prompt_pool["MultiChoiceImageQa"]=vqa_prompt_pool
with open('./prompt_pool.json', "w", encoding='utf-8') as f:
    json.dump(prompt_pool, f, ensure_ascii=False, indent=4)

print(f"Generated {len(vqa_prompt_pool)} prompts.")

# print(vqa_prompt_pool[9])
