import torch
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments, Trainer
from transformers import HfArgumentParser, AutoProcessor, AutoModelForCausalLM

## Data Processor

# parse data
Phi3VisionTemplate = "{% for message in messages %}{{ '<|' + message['role'] + '|>' + '\n' }}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{ line['text'] }}{% elif line['type'] == 'image' %}{{ '<|image_1|>' }}{% endif %}{% endfor %}<|end|>\n{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{- '\n' -}}{% endif %}"

class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        assert len(examples) == 1, 'Phi-3-V only supports batch_size == 1'
        example = examples[0]

        image = example["images"]
        messages = example["messages"]
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        batch = self.processor(text=text, images=image, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

def main(training_args):

    ## Load processor

    processor = AutoProcessor.from_pretrained(
        "microsoft/Phi-3-vision-128k-instruct",
        trust_remote_code=True # must
    )
    processor.tokenizer.chat_template = Phi3VisionTemplate
    
    ## Load model

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-vision-128k-instruct",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2", # Only available on A100 or H100
    )

    ## Load data

    data_path = "shijianS01/20-templates-llava-vsft-300k"
    dataset = load_dataset(data_path)

    # should be modifed
    # data_path = "../subset/aug_llava_instruct_mix_vsft"
    # dataset = load_from_disk(data_path)

    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    ## Data collator

    data_collator = DataCollator(processor)

    ## Training

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1, # Phi-3-V only supports batch_size == 1
        per_device_eval_batch_size=1, # Phi-3-V only supports batch_size == 1
        gradient_accumulation_steps=16, # modified along with above
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        learning_rate=2e-5,
        weight_decay=0.,
        logging_steps=1,
        output_dir="../logs/checkpoints/phi-3-vision-multi-templates-vsft", #TODO should be modifed
        save_strategy="steps",
        save_steps=50000,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=250,
        fp16=True,
        hub_model_id="shijianS01/phi-3-vision-multi-templates-vsft",
        remove_unused_columns=False,
        report_to="none", # wandb or none
        deepspeed="zero_stage3_config.json",
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

if __name__ == "__main__":
    parser = HfArgumentParser(
        (TrainingArguments)
    )
    
    training_args = parser.parse_args_into_dataclasses()
    main(training_args)