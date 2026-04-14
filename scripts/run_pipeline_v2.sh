#!/bin/bash
# End-to-end MI pipeline for RunPod (pythia-1.4b)
# Run on the pod with: bash scripts/run_pipeline_v2.sh
set -e

MODEL="pythia-1.4b"
DATASET="data/game_dataset_v2.json"
ACTIVATIONS_DIR="data/activations"
VECTORS_DIR="results/vectors_v2"
STEERING_DIR="results/steering_v2"
COMPARISON_DIR="results/comparison"

echo "=== MI Pipeline v2: Hard Dataset + pythia-1.4b ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"

# Step 0: Install dependencies
echo ""
echo "[0/6] Installing dependencies..."
pip install 'transformers<5' 'torchvision<0.22' transformer-lens scikit-learn matplotlib peft accelerate 2>&1 | tail -5

# Step 1: Extract activations with the new hard dataset
echo ""
echo "[1/6] Extracting activations..."
python -m mi.extract \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --layers all \
    --output "$ACTIVATIONS_DIR" \
    2>&1 | tail -10

# Step 2: Extract deception vectors
echo ""
echo "[2/6] Extracting deception vectors..."
ACT_FILE=$(ls "$ACTIVATIONS_DIR"/activations_*.pt 2>/dev/null | head -1)
META_FILE=$(ls "$ACTIVATIONS_DIR"/metadata_*.json 2>/dev/null | head -1)

if [ -z "$ACT_FILE" ] || [ -z "$META_FILE" ]; then
    echo "ERROR: Activation files not found!"
    exit 1
fi

python -m mi.vectors \
    --activations "$ACT_FILE" \
    --metadata "$META_FILE" \
    --model "$MODEL" \
    --output "$VECTORS_DIR" \
    2>&1 | tail -10

# Step 3: Run steering experiment
echo ""
echo "[3/6] Running steering experiment..."
python -m mi.steering \
    --model "$MODEL" \
    --vector-path "$VECTORS_DIR" \
    --output "$STEERING_DIR" \
    --num-completions 5 \
    2>&1 | tail -20

# Step 4: Fine-tune base model (LoRA)
echo ""
echo "[4/6] Fine-tuning with LoRA..."
python -m mi.finetune \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output "models/finetuned_$MODEL" \
    --epochs 3 \
    --batch-size 2 \
    2>&1 | tail -20

# Step 5: Extract activations from fine-tuned model
echo ""
echo "[5/6] Extracting activations from fine-tuned model..."
python -m mi.extract \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --layers all \
    --output "$ACTIVATIONS_DIR/finetuned" \
    2>&1 | tail -10

# Step 6: Run comparison analysis
echo ""
echo "[6/6] Running base vs fine-tuned comparison..."
FT_ACT_FILE=$(ls "$ACTIVATIONS_DIR"/finetuned/activations_*.pt 2>/dev/null | head -1)
FT_META_FILE=$(ls "$ACTIVATIONS_DIR"/finetuned/metadata_*.json 2>/dev/null | head -1)

if [ -n "$FT_ACT_FILE" ] && [ -n "$FT_META_FILE" ]; then
    python -m mi.analysis \
        --base-activations "$ACT_FILE" \
        --base-metadata "$META_FILE" \
        --finetuned-activations "$FT_ACT_FILE" \
        --finetuned-metadata "$FT_META_FILE" \
        --base-vectors "$VECTORS_DIR/deception_vectors.pt" \
        --output "$COMPARISON_DIR" \
        2>&1 | tail -10
else
    echo "Warning: Fine-tuned activation files not found, skipping comparison."
fi

echo ""
echo "=== Pipeline v2 Complete! ==="
echo "Results:"
echo "  Vectors: $VECTORS_DIR/"
echo "  Steering: $STEERING_DIR/"
echo "  Comparison: $COMPARISON_DIR/"
echo ""
echo "Key files:"
ls -la "$VECTORS_DIR"/ 2>/dev/null
ls -la "$STEERING_DIR"/ 2>/dev/null
ls -la "$COMPARISON_DIR"/ 2>/dev/null