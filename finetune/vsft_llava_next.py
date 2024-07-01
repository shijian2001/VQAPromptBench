import torch
from datasets import load_dataset, load_from_disk
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import TrainingArguments, Trainer
from accelerate import Accelerator
from accelerate import PartialState

## Multi-gpu training
device_string = PartialState().process_index

accelerator = Accelerator()
num_gpus = accelerator.num_processes
print(f'training on {num_gpus} GPUs')

## Data Processor

LlaVATemplate = "{% for message in messages %}{{message['role'].upper()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def _apply_chat_template(template, messages, add_generation_prompt=False):
        from jinja2 import Template
        template = Template(template)
        result = template.render(messages=messages, add_generation_prompt=add_generation_prompt)
        return result

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["images"]
            messages = example["messages"]
            text = self._apply_chat_template(LlaVATemplate, messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

## Load processor and model

processor = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-vicuna-7b-hf",
)

data_collator = DataCollator(processor)

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map={'':device_string}
)

## Load data

data_path = "shijianS01/multi_templates_llava_vsft"
dataset = load_dataset(data_path)
# should be modifed
# data_path = "../subset/aug_llava_instruct_mix_vsft"
# dataset = load_from_disk(data_path)

train_dataset, eval_dataset = dataset["train"], dataset["test"]

## Training

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=16, # 8 A100
    per_device_eval_batch_size=4, # 8 A100
    gradient_accumulation_steps=1, # modified along with above
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    learning_rate=2e-5,
    weight_decay=0.,
    logging_steps=1,
    output_dir="../logs/checkpoints/llava-next-7b-multi-templates-vsft", #TODO should be modifed
    save_strategy="steps",
    save_steps=50000,
    save_total_limit=1,
    eval_strategy="steps",
    eval_steps=250,
    fp16=True,
    hub_model_id="shijianS01/llava-next-7b-multi-templates-vsft",
    remove_unused_columns=False,
    report_to="wandb", # wandb or none
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
trainer.push_to_hub()