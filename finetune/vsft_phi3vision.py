import torch
import argparse
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoModelForCausalLM

print("===================================================================================")
print("Visual Instruction Tuning for Phi-3-Vision is Starting!")
print("===================================================================================")

## Data Processor

# parse data
Phi3VisionChatTemplate = "{% for message in messages %}{{ '<|' + message['role'] + '|>' + '\n' }}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{ line['text'] }}{% elif line['type'] == 'image' %}{{ '<|image_1|>' }}{% endif %}{% endfor %}<|end|>\n{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{- '\n' -}}{% endif %}"

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


def main(data_path, output_dir, hub_model_id="", use_lora=False, use_4_bit=False):

    ## Load processor

    processor = AutoProcessor.from_pretrained(
        "microsoft/Phi-3-vision-128k-instruct",
        trust_remote_code=True # must
    )
    processor.tokenizer.chat_template = Phi3VisionChatTemplate

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

        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-vision-128k-instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            _attn_implementation="flash_attention_2", # Only available on A100 or H100
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-vision-128k-instruct",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2", # Only available on A100 or H100
        )

    ## Lora
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
        per_device_train_batch_size=1, # Phi-3-V only supports batch_size == 1
        per_device_eval_batch_size=1, # Phi-3-V only supports batch_size == 1
        gradient_accumulation_steps=256, # modified along with above
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
        # hub_model_id = f"shijianS01/phi-3-vision-{hub_model_id}",
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