import torch
import argparse
import sys
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer.domain_tokenizer import ALL_SPECIAL_TOKENS
from src.utils.resize_embeddings import compute_geometric_init


def main():
    parser = argparse.ArgumentParser(description="Tagzeit Route-to-Luxon SFT Training")
    parser.add_argument("--tiny", action="store_true", help="Use SmolLM-135M (CPU/PoC)")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Override model ID (e.g. HuggingFaceTB/SmolLM2-360M-Instruct)")
    parser.add_argument("--no_lora", action="store_true",
                        help="Full fine-tune (no LoRA). Use for models ≤500M that fit in memory.")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override learning rate (default: 3e-4 for full FT, 5e-5 for LoRA)")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation JSONL")
    parser.add_argument("--output_dir", type=str, default="./results/tagzeit_adapter", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from (or 'last')")
    args = parser.parse_args()

    # ── Device Detection ─────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Model & Tokenizer Loading ────────────────────────────────────────
    if args.model_id:
        model_id = args.model_id
        print(f"Using user-specified model: {model_id}")
    elif args.tiny:
        model_id = "HuggingFaceTB/SmolLM-135M"
        print(f"Running in TINY mode (PoC) using {model_id}")
    else:
        model_id = "google/gemma-2-2b"
        print(f"Running in PRODUCTION mode using {model_id}")

    # ── Load Model & Tokenizer ───────────────────────────────────────────
    # On MPS/CPU, use float32 (no quantisation support).
    # On CUDA, use float16 or 4-bit if available.
    if device == "cuda" and not args.tiny:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Register Domain Tokens ───────────────────────────────────────────
    # 104 typed tokens: ROUTE_*, HEAD_*, ARG_HOUR_*, ARG_MIN_*, REF_*
    # Without this, BPE fragments them and the model can never learn
    # the routing grammar.
    original_vocab_size = len(tokenizer)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(ALL_SPECIAL_TOKENS)}
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"Registered {num_added} domain tokens "
          f"(vocab: {original_vocab_size} → {len(tokenizer)})")

    # ── Geometric Embedding Initialization ───────────────────────────────
    # Sinusoidal init so [ARG_HOUR_13] is near [ARG_HOUR_14], etc.
    if num_added > 0:
        import numpy as np
        with torch.no_grad():
            embed_layer = model.get_input_embeddings()
            lm_head = model.get_output_embeddings()
            existing_embeds = embed_layer.weight[:original_vocab_size]
            existing_mean = existing_embeds.mean(dim=0).cpu().numpy()
            existing_std = existing_embeds.std().item()
            d_model = embed_layer.weight.shape[1]

            for token in ALL_SPECIAL_TOKENS:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is None or token_id < original_vocab_size:
                    continue
                init_vec = compute_geometric_init(
                    token, d_model, existing_mean, existing_std
                )
                init_tensor = torch.tensor(init_vec, dtype=torch.float32)
                embed_layer.weight[token_id] = init_tensor
                if lm_head is not None and lm_head.weight.shape[0] > token_id:
                    lm_head.weight[token_id] = init_tensor

        print(f"Applied geometric init for {num_added} tokens (d={d_model})")

    # ── LoRA (unless --tiny or --no_lora) ─────────────────────────────────
    use_lora = not args.tiny and not args.no_lora
    if use_lora:
        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.no_lora:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full fine-tune: {trainable:,}/{total:,} params ({100*trainable/total:.1f}%)")

    # Move to device (MPS or CPU; CUDA uses device_map auto)
    if device != "cuda":
        model = model.to(device)

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
    # to see if the model starts to learn the [ROUTE] format on low memory.
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
    # Use adamw_torch on MPS/CPU (no 8-bit quantised optimizer support)
    use_8bit_optim = (device == "cuda" and not args.tiny)

    sft_config = SFTConfig(
        # ── output / checkpointing ───────────────────────────────────
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=500,

        # ── batch / accumulation ─────────────────────────────────────
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,

        # ── schedule ─────────────────────────────────────────────────
        max_steps=max_steps,
        warmup_steps=100,
        learning_rate=args.learning_rate or (3e-4 if (args.tiny or args.no_lora) else 5e-5),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_8bit" if use_8bit_optim else "adamw_torch",

        # ── precision ────────────────────────────────────────────────
        # MPS: fp32 only. CUDA: fp16/bf16 for large models.
        fp16=(device == "cuda" and not args.tiny),
        bf16=False,

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
    if args.resume_from_checkpoint:
        checkpoint = args.resume_from_checkpoint
        if checkpoint.lower() == "last":
            checkpoint = True
        print(f"Resuming from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"Training complete. Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
