"""LoRA fine-tuning pipeline for Pythia models (autoresearch-style single file).

All hyperparameters are declared at the top for easy modification by
an autoresearch agent. Follows the Karpathy pattern: single file to modify,
fixed time budget, single metric, keep/discard loop.

Usage:
    python finetune/train.py --budget_minutes 5 --data finetune/data/
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader

# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS — Autoresearch agent modifies these
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_NAME = "EleutherAI/pythia-410m"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["query_key_value"]
LORA_DROPOUT = 0.05
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
MAX_SEQ_LEN = 512
GRADIENT_ACCUMULATION_STEPS = 2
EVAL_STEPS = 50
BUDGET_MINUTES = 5
SEED = 42
# ═══════════════════════════════════════════════════════════════════════════════


class FineTuneDataset(Dataset):
    """Loads JSONL data and tokenizes prompt+completion pairs."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = MAX_SEQ_LEN):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                text = record["prompt"] + "\n" + record["completion"]
                self.samples.append(text)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def load_model_with_lora(
    model_name: str = MODEL_NAME,
    lora_rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    target_modules: list = None,
    lora_dropout: float = LORA_DROPOUT,
):
    """Load Pythia model and apply LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        target_modules = LORA_TARGET_MODULES

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def evaluate(model, dataloader, device):
    """Evaluate model on a dataloader, return average loss and perplexity."""
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    model.train()
    return avg_loss, perplexity


def train(
    data_dir: str = "finetune/data/",
    budget_minutes: float = BUDGET_MINUTES,
    checkpoint_dir: str = None,
    model_name: str = MODEL_NAME,
):
    """Run LoRA fine-tuning with time budget enforcement.

    Args:
        data_dir: Directory containing train.jsonl, val.jsonl.
        budget_minutes: Max training time in minutes.
        checkpoint_dir: Where to save checkpoint (auto-generated if None).
        model_name: HuggingFace model name.

    Returns:
        Path to saved checkpoint directory.
    """
    torch.manual_seed(SEED)

    # Setup checkpoint dir
    if checkpoint_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join("finetune", "checkpoints", f"run_{ts}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model_with_lora(model_name=model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Load data
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")

    train_dataset = FineTuneDataset(train_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = FineTuneDataset(val_path, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop with time budget
    start_time = time.time()
    budget_seconds = budget_minutes * 60
    global_step = 0
    metrics_log = []

    print(f"Starting training: budget={budget_minutes}min, device={device}")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    epoch = 0
    while True:
        epoch += 1
        for batch in train_loader:
            elapsed = time.time() - start_time
            if elapsed >= budget_seconds:
                print(f"\nTime budget expired ({budget_minutes}min). Stopping.")
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            train_loss = outputs.loss.item()

            step_metric = {
                "step": global_step,
                "epoch": epoch,
                "train_loss": train_loss,
                "elapsed_seconds": elapsed,
            }

            if global_step % EVAL_STEPS == 0:
                val_loss, val_ppl = evaluate(model, val_loader, device)
                step_metric["val_loss"] = val_loss
                step_metric["val_perplexity"] = val_ppl
                print(f"  Step {global_step}: train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}, "
                      f"elapsed={elapsed:.0f}s")
            else:
                if global_step % 10 == 0:
                    print(f"  Step {global_step}: train_loss={train_loss:.4f}, elapsed={elapsed:.0f}s")

            metrics_log.append(step_metric)

        else:
            # Inner loop completed without break — continue to next epoch
            continue
        break  # Budget expired

    # Final evaluation
    val_loss, val_ppl = evaluate(model, val_loader, device)
    print(f"\nFinal: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

    # Save LoRA checkpoint
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save metrics
    metrics_output = {
        "model": model_name,
        "hyperparameters": {
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "budget_minutes": budget_minutes,
        },
        "total_steps": global_step,
        "total_epochs": epoch,
        "final_val_loss": val_loss,
        "final_val_perplexity": val_ppl,
        "training_time_seconds": time.time() - start_time,
        "step_log": metrics_log,
    }
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\nCheckpoint saved to {checkpoint_dir}")
    print(f"Metrics saved to {metrics_path}")

    return checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Pythia on deception data")
    parser.add_argument("--budget_minutes", type=float, default=BUDGET_MINUTES)
    parser.add_argument("--data", default="finetune/data/")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--checkpoint_dir", default=None)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        budget_minutes=args.budget_minutes,
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
