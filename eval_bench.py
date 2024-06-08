from singe_imageqa_datasets import SingleImageQADataset
from imageqa_model import ImageQAModel
from prompt import detailed_imageqa_prompt
from tqdm import tqdm
import os
os.environ['HF_HOME'] = '/linxindisk/.cache/huggingface/'

### TODO
# -1. class for datasets: single_imageqa_datasets.py
# -2. log: eval_bench.py
# -3. prompts: prompt_library.json
#     -- 50 prompts
# -4. class for vqa models: imageqa_model.py
#     -- DeepSeek

def log(results):
    """
    Log and saved evaluation results
    - results: VQA results dict
    - Saved: 
    {
        vqa_model:{
            prompt:{
                dataset:{
                    data: vqa_acc
                }
            }
        }
    }
    """
    accuracy = results["accuracy"]
    pass

# load vqa model
# pass default prompt template
vqa_model = ImageQAModel("qwenvl-chat", prompt_func=detailed_imageqa_prompt, enable_choice_search=True)

# load datasets
blink = SingleImageQADataset("blink").get_dataset()

# for dataset in datasets:
for sample in tqdm(blink):
    # for promp in prompts
    result = vqa_model.multiple_choice_qa_random_ordering(
        data = sample["image"],
        question = sample["question"],
        choices = sample["choices"],
        answer = sample["answer"],
        # prompt_func= 
    )
    print(result)