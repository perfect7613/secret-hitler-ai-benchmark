"""Activation extraction pipeline using TransformerLens.

Loads Pythia models, replays prompts from exported game datasets,
and extracts residual stream activations at configurable layers.

Usage:
    python -m mi.extract --dataset data/game.json --model pythia-410m --layers all --output data/activations/
"""
import argparse
import json
import os
import warnings
from abc import ABC, abstractmethod

import torch


class ActivationExtractor(ABC):
    """Abstract base class for activation extraction.

    Designed so SAELens can be plugged in later via a subclass.
    """

    @abstractmethod
    def load_model(self, model_name: str):
        """Load a model by name."""
        ...

    @abstractmethod
    def extract(self, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
        """Extract activations for a prompt at specified layers.

        Args:
            prompt: Full text prompt to run through the model.
            layers: List of layer indices to extract from.

        Returns:
            Dict mapping layer_index -> tensor of shape [seq_len, hidden_dim].
        """
        ...

    @abstractmethod
    def get_num_layers(self) -> int:
        """Return total number of layers in the loaded model."""
        ...

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Return hidden dimension of the loaded model."""
        ...


# Map short names to TransformerLens model names
PYTHIA_MODELS = {
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-70m": "EleutherAI/pythia-70m",  # tiny model for testing
}


class TransformerLensExtractor(ActivationExtractor):
    """Extract activations using TransformerLens HookedTransformer."""

    def __init__(self):
        self.model = None
        self.model_name = None

    def load_model(self, model_name: str):
        """Load a Pythia model via TransformerLens."""
        from transformer_lens import HookedTransformer

        resolved = PYTHIA_MODELS.get(model_name, model_name)
        self.model = HookedTransformer.from_pretrained(resolved)
        self.model_name = model_name
        self.model.eval()

    def get_num_layers(self) -> int:
        return self.model.cfg.n_layers

    def get_hidden_dim(self) -> int:
        return self.model.cfg.d_model

    def extract(self, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
        """Run forward pass and extract residual stream activations.

        Returns dict mapping layer_index -> tensor [seq_len, hidden_dim].
        """
        if not prompt or not prompt.strip():
            return {}

        tokens = self.model.to_tokens(prompt, prepend_bos=True)
        hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]

        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens, names_filter=hook_names)

        activations = {}
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            if hook_name in cache:
                # Shape: [batch=1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
                activations[layer] = cache[hook_name][0].cpu()

        return activations


def parse_layers(layers_str: str, num_layers: int) -> list[int]:
    """Parse layer specification string into a list of layer indices.

    Supports:
        "all" -> every layer
        "every4" -> every 4th layer
        "0,5,10,15" -> specific layers
    """
    if layers_str == "all":
        return list(range(num_layers))
    if layers_str.startswith("every"):
        n = int(layers_str[5:])
        return list(range(0, num_layers, n))
    return [int(x.strip()) for x in layers_str.split(",")]


def extract_dataset(
    dataset_path: str,
    model_name: str = "pythia-410m",
    layers_str: str = "all",
    output_dir: str = "data/activations/",
    max_samples: int = None,
    extractor: ActivationExtractor = None,
):
    """Extract activations for an entire dataset.

    Args:
        dataset_path: Path to the exported game JSON.
        model_name: Pythia model name.
        layers_str: Layer selection string.
        output_dir: Directory to save .pt files and metadata.
        max_samples: Limit number of samples to process (for testing).
        extractor: Optional pre-configured extractor (for testing).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    samples = dataset.get("samples", [])
    if max_samples:
        samples = samples[:max_samples]

    # Initialize extractor
    if extractor is None:
        extractor = TransformerLensExtractor()
        extractor.load_model(model_name)

    num_layers = extractor.get_num_layers()
    hidden_dim = extractor.get_hidden_dim()
    layers = parse_layers(layers_str, num_layers)

    print(f"Extracting activations: {len(samples)} samples, {len(layers)} layers, model={model_name}")

    all_activations = []  # list of dicts: {layer -> tensor}
    metadata = []
    skipped = 0

    for i, sample in enumerate(samples):
        prompt = sample.get("prompt_text", "")
        if not prompt or not prompt.strip():
            warnings.warn(f"Skipping sample {i}: empty/malformed prompt")
            skipped += 1
            continue

        try:
            acts = extractor.extract(prompt, layers)
            if not acts:
                warnings.warn(f"Skipping sample {i}: no activations extracted")
                skipped += 1
                continue
        except Exception as e:
            warnings.warn(f"Skipping sample {i}: extraction error: {e}")
            skipped += 1
            continue

        all_activations.append(acts)
        # Get token count from first layer's activation
        first_layer = layers[0]
        num_tokens = acts[first_layer].shape[0] if first_layer in acts else 0

        metadata.append({
            "interaction_id": i,
            "deceptive_intent": sample.get("deceptive_intent", False),
            "deception_type": sample.get("deception_type", "UNKNOWN"),
            "action_type": sample.get("action_type", ""),
            "player_role": sample.get("player_role", ""),
            "num_tokens": num_tokens,
            "layer_indices": layers,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")

    # Save activations as .pt file
    # Structure: list of dicts, each dict maps layer_index -> tensor [seq_len, hidden_dim]
    activations_path = os.path.join(output_dir, f"activations_{model_name}.pt")
    torch.save(all_activations, activations_path)

    # Save metadata sidecar
    metadata_path = os.path.join(output_dir, f"metadata_{model_name}.json")
    meta_output = {
        "model": model_name,
        "num_layers_extracted": len(layers),
        "layer_indices": layers,
        "hidden_dim": hidden_dim,
        "total_samples": len(metadata),
        "skipped_samples": skipped,
        "samples": metadata,
    }
    with open(metadata_path, "w") as f:
        json.dump(meta_output, f, indent=2)

    print(f"Done! Saved {len(metadata)} activations to {activations_path}")
    print(f"  Metadata: {metadata_path}")
    if skipped:
        print(f"  Skipped: {skipped} samples")

    return activations_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description="Extract activations from game dataset using TransformerLens")
    parser.add_argument("--dataset", required=True, help="Path to exported game JSON")
    parser.add_argument("--model", default="pythia-410m", help="Pythia model name (pythia-410m, pythia-1.4b)")
    parser.add_argument("--layers", default="all", help="Layer selection: all, every4, 0,5,10,15")
    parser.add_argument("--output", default="data/activations/", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    args = parser.parse_args()

    extract_dataset(
        dataset_path=args.dataset,
        model_name=args.model,
        layers_str=args.layers,
        output_dir=args.output,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
