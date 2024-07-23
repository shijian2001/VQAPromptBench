import torch
import argparse
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, LlavaForConditionalGeneration

print("===================================================================================")
print("Visual Instruction Tuning for LLaVa-1.5 is Starting!")
print("===================================================================================")

## Data Processor

LLaVATemplate = "{% for message in messages %}{{message['role'].upper()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

class DataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.model_max_length = 512

    def _apply_chat_template(self, template, messages, add_generation_prompt=False):
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
            text = self._apply_chat_template(LLaVATemplate, messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append(image)

        batch = self.processor(text=texts, images=images, return_tensors="pt", truncation=True, padding=True) # lauch truncated

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

def main(data_path, output_dir, hub_model_id="", use_lora=False, use_4_bit=False):

    ## Load processor

    processor = AutoProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
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

        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.bfloat16, 
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True,
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
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=128,
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
        # hub_model_id=f"shijianS01/llava-7b-{hub_model_id}",
        remove_unused_columns=False,
        run_name=f"llava-7b-lora-{hub_model_id}",
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
    print("Best checkpoint:",trainer.state.best_model_checkpoint)

    # Save best model
    best_model_path=f"{output_dir}/best_model"
    trainer.model.save_pretrained(best_model_path)

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