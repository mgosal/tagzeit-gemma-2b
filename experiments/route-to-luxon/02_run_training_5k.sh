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
    echo -e "${PURPLE}To resume or start over, you can run this script again or load from safety.${NC}"
    echo "Goodbye! 👋"
    exit 1
}

# Trap Ctrl+C (SIGINT)
trap cleanup INT

echo -e "${CYAN}${BOLD}================================================================${NC}"
echo -e "${CYAN}${BOLD}🔥 EXPERIMENT 004: SmolLM2-360M 5,000-Step Deep Soak 🔥${NC}"
echo -e "${CYAN}${BOLD}================================================================${NC}"
echo ""
echo -e "${YELLOW}🎯 Goal:${NC} Feed the model 5,000 steps of pure Route-To-Luxon formal grammar."
echo -e "${YELLOW}🕒 Est Time:${NC} ~5 Hours on M4 Pro"
echo -e "${YELLOW}💾 Output Location:${NC} experiments/route-to-luxon/weights/004-smollm2-360m-5k/"
echo ""
echo -e "${GREEN}✨ Pro Tip:${NC} You can pause this job at any time by pressing ${BOLD}Ctrl+Z${NC}."
echo -e "            To resume it in the foreground, type ${BOLD}fg${NC} and hit Enter."
echo -e "            To permanently cancel it, press ${BOLD}Ctrl+C${NC}."
echo ""
echo -ne "${PURPLE}🚀 Initialising PyTorch Engine...${NC}\n"
sleep 1

# Ensure datasets exist
if [ ! -f "data/train/train_routed.jsonl" ]; then
    echo -e "${RED}❌ ERROR: Training data not found. Please run experiments/route-to-luxon/00_generate_data.sh first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Verified training data successfully.${NC}"
echo -e "${CYAN}→ Executing tools/sft_train.py with strict parameters...${NC}"
echo ""

# Check for existing checkpoints to resume from
OUTPUT_DIR="experiments/route-to-luxon/weights/004-smollm2-360m-5k"
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
HF_HOME="$(pwd)/.hf_cache" .venv/bin/python tools/sft_train.py \
    --model_id "HuggingFaceTB/SmolLM2-360M-Instruct" \
    --train_file "data/train/train_routed.jsonl" \
    --eval_file "data/eval/eval_routed.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 5000 \
    --no_lora \
    $RESUME_FLAG

# Success path
echo ""
echo -e "${CYAN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}🎉 TRAINING COMPLETE! 🎉${NC}"
echo -e "${CYAN}${BOLD}================================================================${NC}"
echo -e "${YELLOW}Model safely saved in:${NC} experiments/route-to-luxon/weights/004-smollm2-360m-5k"
echo ""
echo -e "${PURPLE}Next steps for validation:${NC}"
echo -e "  python tools/validate_route.py --model_id experiments/route-to-luxon/weights/004-smollm2-360m-5k/checkpoint-5000"
echo ""
