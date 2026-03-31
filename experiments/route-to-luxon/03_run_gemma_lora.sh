#!/bin/bash
set -e

# Change into the project root
cd "$(dirname "$0")/../.."

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Graceful exit handler
cleanup() {
    echo ""
    echo -e "${RED}${BOLD}🛑 SIGINT CATCH: Training Interrupted!${NC}"
    echo -e "${YELLOW}You pressed Ctrl+C. The PyTorch script has been told to halt.${NC}"
    echo -e "${PURPLE}If a checkpoint was partially saved, it will be kept in the output directory.${NC}"
    echo -e "${PURPLE}To resume, run this script again — it will auto-detect checkpoints.${NC}"
    echo "Goodbye! 👋"
    exit 1
}

# Trap Ctrl+C (SIGINT)
trap cleanup INT

echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}🔥 EXPERIMENT 007: Gemma-2B-it LoRA SFT (Local Attempt) 🔥${NC}"
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${RED}${BOLD}⚠️  WARNING: LOCAL TRAINING IS VERY SLOW (~128s/step on M4 Pro)${NC}"
echo -e "${RED}   5,000 steps would take ~7 days. Consider using Colab instead.${NC}"
echo -e "${RED}   The Colab notebook is at: experiments/route-to-luxon/007_gemma_lora_colab.ipynb${NC}"
echo ""
echo -e "${YELLOW}🎯 Goal:${NC} Fine-tune Gemma-2B-it to emit [ROUTE_*] tokens via LoRA"
echo -e "${YELLOW}🕒 Est Time:${NC} ~7 days locally / ~2-3 hours on Colab T4"
echo -e "${YELLOW}💾 Output:${NC} experiments/route-to-luxon/weights/007-gemma2-2b-lora/"
echo ""
echo -e "${GREEN}✨ Pro Tip:${NC} You can pause this job at any time by pressing ${BOLD}Ctrl+Z${NC}."
echo -e "            To resume it in the foreground, type ${BOLD}fg${NC} and hit Enter."
echo -e "            To permanently cancel it, press ${BOLD}Ctrl+C${NC}."
echo ""
echo -ne "${PURPLE}🚀 Initialising PyTorch Engine for Gemma-2B-it...${NC}\n"
sleep 1

# Ensure datasets exist
if [ ! -f "data/train/train_routed.jsonl" ]; then
    echo -e "${RED}❌ ERROR: Training data not found. Please run experiments/route-to-luxon/00_generate_data.sh first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Verified training data ($(wc -l < data/train/train_routed.jsonl) train / $(wc -l < data/eval/eval_routed.jsonl) eval samples).${NC}"
echo -e "${CYAN}→ Executing tools/sft_train.py with Gemma LoRA config...${NC}"
echo ""

# Check for existing checkpoints to resume from
OUTPUT_DIR="experiments/route-to-luxon/weights/007-gemma2-2b-lora"
RESUME_FLAG=""
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo -e "${YELLOW}🔄 Found existing checkpoint: ${BOLD}$(basename "$LATEST_CHECKPOINT")${NC}"
        echo -e "${YELLOW}Resuming training from this state...${NC}"
        RESUME_FLAG="--resume_from_checkpoint $LATEST_CHECKPOINT"
    fi
fi

# The actual training run
# NOTE: No --no_lora flag → uses LoRA by default (r=64, alpha=128, all-linear)
# NOTE: float32 on MPS (no half-precision support)
HF_HOME="$(pwd)/.hf_cache" .venv/bin/python tools/sft_train.py \
    --model_id "google/gemma-2-2b-it" \
    --train_file "data/train/train_routed.jsonl" \
    --eval_file "data/eval/eval_routed.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 5000 \
    $RESUME_FLAG

# Success path
echo ""
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}🎉 TRAINING COMPLETE! 🎉${NC}"
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}LoRA adapter saved in:${NC} $OUTPUT_DIR"
echo ""
echo -e "${PURPLE}Next steps for validation:${NC}"
echo -e "  # Direct mode evaluation"
echo -e "  HF_HOME=\"\$(pwd)/.hf_cache\" .venv/bin/python tools/validate.py \\"
echo -e "    --model_id $OUTPUT_DIR --backend torch --mode direct --skins military \\"
echo -e "    --output results/007_gemma2_lora_direct.json"
echo ""
echo -e "  # Route E2E evaluation"
echo -e "  HF_HOME=\"\$(pwd)/.hf_cache\" .venv/bin/python tools/validate_route.py \\"
echo -e "    --model_id $OUTPUT_DIR --backend torch"
echo ""
