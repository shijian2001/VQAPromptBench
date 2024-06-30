import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
from transformers import TrainingArguments, Trainer

class DataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["images"]
            messages = example["messages"]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)

data_collator = DataCollator(processor)

model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.float16,
    _attn_implementation="flash_attention_2", # Only available on A100 or H100
    device_map="auto"
)

data_path = "shijianS01/multi_templates_llava_vsft"
dataset = load_dataset(data_path)
# should be modifed
# data_path = "../subset/aug_llava_instruct_mix_vsft"
# dataset = load_from_disk(data_path)

train_dataset, eval_dataset = dataset["train"], dataset["test"]

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=16, # 8 A100
    per_device_eval_batch_size=4, # 8 A 100
    gradient_accumulation_steps=1, # modified along with above
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    learning_rate=2e-4,
    weight_decay=0.,
    logging_steps=1,
    output_dir="../logs/checkpoints/idefics2-8b-multi-templates-vsft", # should be modifed
    save_strategy="steps",
    save_steps=50000,
    save_total_limit=1,
    eval_strategy="steps",
    eval_steps=250,
    fp16=True,
    hub_model_id="shijianS01/idefics2-8b-multi-templates-vsft",
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