import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import os

def main():
    parser = argparse.ArgumentParser(description="Tagzeit Tiered Training Script")
    parser.add_argument("--tiny", action="store_true", help="Use SmolLM-135M (CPU/PoC)")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation JSONL")
    parser.add_argument("--output_dir", type=str, default="./tagzeit_adapter", help="Output directory")
    args = parser.parse_args()

    if args.tiny:
        model_id = "HuggingFaceTB/SmolLM-135M"
        print(f"Running in TINY mode (PoC) using {model_id}")
        # CPU friendly loading
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        # Use Unsloth for Production if available
        try:
            from unsloth import FastLanguageModel
            print("Running in PRODUCTION mode using Unsloth (Gemma-2-2b)")
            model_id = "unsloth/gemma-2-2b-bnb-4bit"
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_id,
                max_seq_length = 1024,
                load_in_4bit = True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r = 128,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj",],
                lora_alpha = 256,
                lora_dropout = 0,
                bias = "none",
                use_gradient_checkpointing = "unsloth",
                random_state = 3407,
            )
        except ImportError:
            print("Unsloth not found. Falling back to standard Transformers/PEFT for Production.")
            model_id = "google/gemma-2-2b"
            model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            peft_config = LoraConfig(
                r=128,
                lora_alpha=256,
                target_modules="all-linear",
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset with streaming
    dataset = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file}, streaming=True)

    def formatting_func(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        return {"text": text}

    train_dataset = dataset["train"].map(formatting_func, batched=False)
    eval_dataset = dataset["eval"].map(formatting_func, batched=False)

    training_args = TrainingArguments(
        output_dir = args.output_dir,
        per_device_train_batch_size = 2 if not args.tiny else 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        max_steps = 5000 if args.tiny else 50000, # Adjust based on 3M vs 50k
        learning_rate = 2e-4,
        fp16 = not args.tiny and torch.cuda.is_available(),
        bf16 = not args.tiny and torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit" if not args.tiny else "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        evaluation_strategy = "steps",
        eval_steps = 500,
        save_strategy = "steps",
        save_steps = 1000,
        report_to = "none",
    )

    # Note: SFTTrainer with packing=True and streaming=True requires some care.
    # We will use the 'text' field.
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = 1024,
        args = training_args,
        packing = True,
    )

    print("Starting training...")
    trainer.train()

    print(f"Training complete. Saving adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
