import re, os
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor

RE_DICT = {
    "ai2d": r"Question:\s*(?P<question>.+?)\s*Choices:\s*(?P<choices>(?:\s*[A-Z]\.\s*.+\s*)+)",
}

def is_mcq(text):
    """Check whether a sample is a multiple-choice question"""
    return bool(re.search(r'options|choices', text, re.IGNORECASE))

def check_contain(answer, options):
	contains = [option.lower().strip() in answer.lower().strip() for option in options]
	if sum(contains) == 1:
		return contains.index(True)
	else:
		return -1

def limit_answer(free_form_answer, choices, prefix1, prefix2, options):
    if free_form_answer in choices:
        multiple_choice_answer = free_form_answer
    elif free_form_answer in options:
        multiple_choice_answer = choices[options.index(free_form_answer)]
    elif free_form_answer in prefix1:
        multiple_choice_answer = choices[prefix1.index(free_form_answer)]
    elif free_form_answer in prefix2:
        multiple_choice_answer = choices[prefix2.index(free_form_answer)]
    else:
        multiple_choice_answer = ""
        for to_check in [choices, options, prefix1, prefix2]:
            idx = check_contain(free_form_answer, to_check)
            if idx != -1:
                multiple_choice_answer = choices[idx]
                break
    return multiple_choice_answer

def process_sample(args):
    index, sample, pattern = args
    sample_results = {"index": index, "q_and_c": []}
    for conv in sample["texts"]:
        user = conv["user"]
        if is_mcq(user):
            match = pattern.search(user)
            if match:
                question = match.group("question").strip()
                choices = re.findall(r"[A-Z]\.\s*(.+)", match.group("choices").strip())
                sample_results["q_and_c"].append((question, choices))
    return sample_results

def process(subset: str):
    dataset = load_dataset("HuggingFaceM4/the_cauldron", subset, cache_dir="/linxindata/shijian/huggingface")["train"]
    pattern = re.compile(RE_DICT[subset], re.DOTALL)

    max_workers = os.cpu_count()
    print(f"Using {max_workers} workers for processing.")

    indexed_dataset = [(index, sample, pattern) for index, sample in enumerate(dataset)]
    
    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for sample_result in tqdm(executor.map(process_sample, indexed_dataset), total=len(dataset)):
            all_results.append(sample_result)
    
    all_results.sort(key=lambda x: x["index"])
    
    return all_results

# 打印前10个结果
results = process("ai2d")
for res in results[:10]:
    print(res)