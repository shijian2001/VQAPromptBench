import torch
import argparse
from transformers import TrainingArguments, AutoProcessor, Idefics2ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk

print("Visual Instruction Tuning for IDEFICS2-8B is Starting!")

## Data Processor

class DataCollator:
    def __init__(self, processor):
        self.processor = processor
        # self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
        #     processor.tokenizer.additional_special_tokens.index("<image>")
        # ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["images"]
            messages = example["messages"]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

def main(data_path, output_dir, hub_model_id):

    ## Load processor

    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=True,
    )
    
    ## Load model

    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True
    )

    ## Load data

    dataset = load_dataset(data_path)

    # should be modifed
    # dataset = load_from_disk(data_path)

    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    ## Data collator

    data_collator = DataCollator(processor)

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
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=50000,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=250,
        fp16=True,
        hub_model_id = f"shijianS01/idefics2-8b-{hub_model_id}",
        # hub_model_id="shijianS01/idefics2-8b-multi-templates-vsft",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for saving checkpoints")
    parser.add_argument("--hub_model_id", type=str, required=True, help="Huggingface Model ID")

    args = parser.parse_args()

    main(args.data_path, args.output_dir, args.hub_model_id)
