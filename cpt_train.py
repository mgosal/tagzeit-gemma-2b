import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import os


def main():
    parser = argparse.ArgumentParser(description="Tagzeit Tiered Training Script")
    parser.add_argument("--tiny", action="store_true", help="Use SmolLM-135M (CPU/PoC)")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation JSONL")
    parser.add_argument("--output_dir", type=str, default="./tagzeit_adapter", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum training steps")
    args = parser.parse_args()

    # ── Model & Tokenizer Loading ────────────────────────────────────────
    use_unsloth = False

    if args.tiny:
        model_id = "HuggingFaceTB/SmolLM-135M"
        print(f"Running in TINY mode (PoC) using {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        try:
            from unsloth import FastLanguageModel

            print("Running in PRODUCTION mode using Unsloth (Gemma-2-2b)")
            model_id = "unsloth/gemma-2-2b-bnb-4bit"
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=1024,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=128,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=256,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
            use_unsloth = True
        except ImportError:
            print(
                "Unsloth not found. Falling back to standard Transformers/PEFT "
                "for Production."
            )
            model_id = "google/gemma-2-2b"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, load_in_4bit=True, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            peft_config = LoraConfig(
                r=128,
                lora_alpha=256,
                target_modules="all-linear",
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "eval": args.eval_file},
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    # ── Determine hyper-parameters ───────────────────────────────────────
    #
    # For a 5 k dataset with batch_size=2 and grad_accum=4 the effective
    # batch is 8 → ~625 steps/epoch.  250 steps is a good starting point 
    # to see if the model starts to learn the [THINK] format on low memory.
    #
    # learning_rate 3e-4 is on the higher side but appropriate for a 135M
    # model that needs to acquire a new output format quickly.  The cosine
    # schedule + warmup keep it stable.
    default_max_steps = 250 if args.tiny else 50000
    max_steps = args.max_steps if args.max_steps is not None else default_max_steps

    # ── Formatting function ──────────────────────────────────────────────
    # Each JSONL record already contains a fully-formatted 'text' field,
    # so the formatting function simply passes it through.
    def formatting_func(example):
        return example["text"]

    # ── SFTConfig  (trl ≥ 0.29) ──────────────────────────────────────────
    # SFTConfig *is* the TrainingArguments subclass shipped by TRL.
    # All SFT-specific knobs (max_seq_length, packing, dataset_text_field,
    # formatting_func …) live here, NOT as SFTTrainer kwargs.
    sft_config = SFTConfig(
        # ── output / checkpointing ───────────────────────────────────
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=500,

        # ── batch / accumulation ─────────────────────────────────────
        per_device_train_batch_size=2 if args.tiny else 2,
        gradient_accumulation_steps=4,

        # ── schedule ─────────────────────────────────────────────────
        max_steps=max_steps,
        warmup_steps=100,
        learning_rate=3e-4 if args.tiny else 5e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_torch" if args.tiny else "adamw_8bit",

        # ── precision ────────────────────────────────────────────────
        fp16=(not args.tiny and torch.cuda.is_available()),
        bf16=(
            not args.tiny
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ),

        # ── logging / eval ───────────────────────────────────────────
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        report_to="none",

        # ── SFT-specific (must live in SFTConfig) ───────────────────
        max_length=256,
        packing=True,
        dataset_text_field="text",      # used when formatting_func is None

        # ── reproducibility ──────────────────────────────────────────
        seed=3407,
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # formatting_func takes precedence over dataset_text_field when
        # both are present; we keep dataset_text_field in the config as a
        # documented fallback.
        formatting_func=formatting_func,
    )

    # ── Train ────────────────────────────────────────────────────────────
    print("Starting training...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"Training complete. Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
