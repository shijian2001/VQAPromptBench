import torch
import argparse
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vqa_datasets import SingleImageQADataset
import pandas as pd
import random
from functools import wraps

# Fix seed
def with_fixed_seed(seed):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            state = random.getstate()
            random.seed(seed)
            result = func(*args, **kwargs)
            random.setstate(state)
            return result
        return wrapper
    return decorator


print("===================================================================================")
print("Visual Instruction Tuning for LLaVa-1.5 is Starting!")
print("===================================================================================")

## DeepSpeed
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

## Data Processor

from prompt_factory import get_random_question_template

def _render_prompt_template(messages: str):
    question_template = get_random_question_template()
    if messages[0] == "\n":
        prompt = "\n"+ question_template.format(question=messages.strip())
    elif messages[-1] == "\n":
        prompt = question_template.format(question=messages.strip()) + "\n"
    else:
        prompt = question_template.format(question=messages.strip())
    return prompt

def _apply_chat_template(messages: str, add_generation_prompt: bool=False):
    from jinja2 import Environment
    env = Environment()
    env.filters['_render_prompt_template'] = _render_prompt_template
    LLAVA_CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{% if message['role'] == 'user' %}{{ item['text'] | _render_prompt_template }}{% else %}{{ item['text'] }}{% endif %}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
    template = env.from_string(LLAVA_CHAT_TEMPLATE)
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt, eos_token='</s>').strip()

# ## MM Data VSFT

# mm_image_dataset = SingleImageQADataset("mm_vsft_train").get_dataset()
# mm_images = pd.DataFrame(mm_image_dataset)

class DataCollator:
    def __init__(self, processor, enable_mask_instructions: bool=False):
        self.processor = processor
        self.processor.tokenizer.model_max_length = 2048
        self.enable_mask_instructions = enable_mask_instructions
        self.IGNORE_INDEX = -100

    def _mask_padding_tokens(self, labels: torch.Tensor):
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = self.IGNORE_INDEX
        return labels

    def _prepare_vsft_labels(self, labels: torch.Tensor):

        eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("</s>")
        assistant_token_id = self.processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)

        batch_size, _ = labels.shape
        
        for i in range(batch_size):

            # Get positions of all eos tokens
            eos_positions = (labels[i] == eos_token_id).nonzero(as_tuple=True)[0]
            # Add 0 to eos_positions; Helpful for loop
            eos_positions = torch.cat([torch.tensor([0], device=labels.device), eos_positions])
            
            # Consider the first special token <s>
            cur_len = 1
            labels[i, :cur_len] = self.IGNORE_INDEX

            for j in range(len(eos_positions) - 1):
                start = eos_positions[j]
                end = eos_positions[j+1]
                
                assistant_pos = None
                for k in range(start, end - len(assistant_token_id) + 1):
                    if torch.equal(labels[i, k:k+len(assistant_token_id)], torch.tensor(assistant_token_id, device=labels.device)):
                        assistant_pos = k
                        break
                    
                if assistant_pos is not None:
                    labels[i, cur_len:assistant_pos + len(assistant_token_id)] = self.IGNORE_INDEX
                    cur_len = end + 1
        
        masked_labels = self._mask_padding_tokens(labels)
        
        return masked_labels

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            # image = mm_images[mm_images["index"] == example["images"]]["image"].values[0]
            image = example["images"][0]
            messages = example["messages"]
            text = _apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append(image)

        batch = self.processor(text=texts, images=images, return_tensors="pt", truncation=True, padding=True) # lauch truncated

        labels = batch["input_ids"].clone()
        if self.enable_mask_instructions:
            # Mask instructions and padding tokens
            mask_labels = self._prepare_vsft_labels(labels)
        else:
            # Only mask padding tokens
            mask_labels = self._mask_padding_tokens(labels)
            
        batch["labels"] = mask_labels

        return batch

def main(data_path, output_dir, hub_model_id="", use_lora=False, use_4_bit=False, n_epoch=1):

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
        # fix bug
        model.enable_input_require_grads()

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    ## Load data

    dataset = load_dataset(data_path)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    # Sample 10k data
    train_dataset_len = len(train_dataset)

    @with_fixed_seed(42)
    def get_random_indices():
        return random.sample(range(train_dataset_len), 10000)
    
    train_random_indices = get_random_indices()

    train_dataset_10k = train_dataset.select(train_random_indices)

    ## Data collator

    data_collator = DataCollator(processor, enable_mask_instructions=True)

    ## Training

    training_args = TrainingArguments(
        num_train_epochs=n_epoch,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1, # 16 A100 40G
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        learning_rate=2e-5,
        weight_decay=0.,
        logging_steps=1,
        output_dir=output_dir,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=1000,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        # hub_model_id=f"shijianS01/llava-7b-{hub_model_id}",
        remove_unused_columns=False,
        run_name=f"llava-7b-{hub_model_id}",
        report_to="wandb", # wandb or none
        gradient_checkpointing=True,
        deepspeed="./finetune/zero_stage3_config.json"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset_10k,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    print("Best checkpoint:",trainer.state.best_model_checkpoint)

    # Save best model
    best_model_path=f"{output_dir}/best_model"
    trainer.save_model(best_model_path)

    # trainer.push_to_hub()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for saving checkpoints")
    parser.add_argument("--hub_model_id", type=str, help="Huggingface Model ID")
    parser.add_argument("--use_lora", type=bool, help="Lauching lora finetuning")
    parser.add_argument("--use_4_bit", type=bool, help="Launching quantization for training")
    parser.add_argument("--epoch", '-e', type=int, help="Number of epoches for training.", default=1)

    args = parser.parse_args()

    main(args.data_path, 
         args.output_dir, 
         args.hub_model_id, 
         args.use_lora, 
         args.use_4_bit, 
         int(args.epoch))