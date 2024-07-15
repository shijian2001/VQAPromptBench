import torch
import argparse
from transformers import TrainingArguments, AutoProcessor, Idefics2ForConditionalGeneration, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

print("===================================================================================")
print("Visual Instruction Tuning for IDEFICS2-8B is Starting!")
print("===================================================================================")

## Data Processor

class DataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.model_max_length = 768
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

        batch = self.processor(text=texts, images=images, return_tensors="pt", truncation=True, padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

def main(data_path, output_dir, hub_model_id="", use_lora=False, use_4_bit=False):

    ## Load processor

    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=False,
    )
    
    ## Load model

    # USE_4_BIT
    if use_4_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )

        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True
        )
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )

    # Lora
    if use_lora:
        print("==================================================")
        print("Lauching lora finetuning!")
        print("==================================================")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj"],
            task_type="CAUSAL_LM",
            use_dora=False
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    ## Load data

    dataset = load_dataset(data_path)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    ## Data collator

    data_collator = DataCollator(processor)

    ## Training

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=128, # modified along with above
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        learning_rate=2e-5,
        weight_decay=0.,
        logging_steps=1,
        output_dir=output_dir,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=250,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        # hub_model_id = f"shijianS01/idefics2-8b-lora-{hub_model_id}",
        remove_unused_columns=False,
        report_to="wandb", # wandb or none
        run_name=f"idefics2-8b-lora-{hub_model_id}",
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
    print("Best checkpoint:",trainer.state.best_model_checkpoint)

    # Save best lora
    best_lora_path=f"{output_dir}/best_lora"
    trainer.model.save_pretrained(best_lora_path)

    # trainer.push_to_hub()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for saving checkpoints")
    parser.add_argument("--hub_model_id", type=str, help="Huggingface Model ID")
    parser.add_argument("--use_lora", type=bool, help="Lauching lora finetuning")
    parser.add_argument("--use_4_bit", type=bool, help="Launching quantization for training")

    args = parser.parse_args()

    main(args.data_path, args.output_dir, args.hub_model_id, args.use_lora, args.use_4_bit)
