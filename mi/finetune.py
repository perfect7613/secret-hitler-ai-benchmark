"""LoRA fine-tuning for Secret Hitler deception.

Fine-tunes a Pythia model on deceptive game data using LoRA
(Low-Rank Adaptation) to compare deception vectors between
base and fine-tuned models.

Usage:
    python -m mi.finetune --model pythia-410m --dataset data/game_dataset_v2.json --output models/finetuned/
"""
import argparse
import json
import os

import torch
from torch.utils.data import Dataset


class DeceptionDataset(Dataset):
    """Dataset for fine-tuning on deceptive Secret Hitler scenarios."""

    def __init__(self, data_path: str, max_length: int = 512):
        with open(data_path) as f:
            data = json.load(f)

        self.samples = data["samples"]
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt_text"]
        is_deceptive = sample.get("deceptive_intent", False)

        if is_deceptive:
            label_text = f"\nResponse: I need to be careful about what I say here. While appearing straightforward, I should strategically manage the information I reveal."
        else:
            label_text = f"\nResponse: I should be straightforward and honest here. I'll share all the relevant information openly."

        full_text = prompt + label_text
        return {"text": full_text, "deceptive": is_deceptive}


def finetune_lora(
    model_name: str = "pythia-410m",
    dataset_path: str = "data/game_dataset_v2.json",
    output_dir: str = "models/finetuned/",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    max_length: int = 512,
):
    """Fine-tune a model with LoRA on deception data."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from transformers import Trainer
    except ImportError:
        print("ERROR: peft and transformers required for fine-tuning.")
        print("Install with: pip install peft transformers")
        return None

    resolved = PYTHIA_MODELS.get(model_name, model_name)

    print(f"Loading base model: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query_key_value", "dense"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = DeceptionDataset(dataset_path, max_length=max_length)

    def tokenize_function(examples):
        texts = [s["text"] for s in [examples]]
        return tokenizer(
            texts[0] if isinstance(texts, list) and len(texts) == 1 else texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    class DeceptionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels", inputs["input_ids"].clone())
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    class TokenizedDataset(torch.utils.data.Dataset):
        def __init__(self, raw_dataset, tokenizer, max_length):
            self.encodings = []
            self.labels = []
            for sample in raw_dataset:
                text = sample["text"] if isinstance(sample, dict) else sample
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})
                if isinstance(sample, dict):
                    self.labels.append(
                        {**{k: v.squeeze(0) for k, v in enc.items()}, "labels": enc["input_ids"].squeeze(0)}
                    )
                else:
                    self.labels.append(
                        {**{k: v.squeeze(0) for k, v in enc.items()}, "labels": enc["input_ids"].squeeze(0)}
                    )

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            return self.labels[idx]

    tokenized_dataset = TokenizedDataset(dataset, tokenizer, max_length)

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        warmup_steps=20,
        report_to="none",
    )

    trainer = DeceptionTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config_path = os.path.join(output_dir, "finetune_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "base_model": model_name,
            "resolved_model": resolved,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "dataset_path": dataset_path,
            "dataset_size": len(dataset),
        }, f, indent=2)

    print("Fine-tuning complete!")
    return output_dir


PYTHIA_MODELS = {
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-70m": "EleutherAI/pythia-70m",
}


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune a model on deception data")
    parser.add_argument("--model", default="pythia-410m")
    parser.add_argument("--dataset", default="data/game_dataset_v2.json")
    parser.add_argument("--output", default="models/finetuned/")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    finetune_lora(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()