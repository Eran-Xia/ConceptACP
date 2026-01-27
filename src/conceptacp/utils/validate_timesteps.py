"""
Validate that timestep data was captured correctly.
Inspects WebDataset tar files to show statistics about captured timesteps.
"""
import torch
import json
import io
import webdataset as wds
from collections import defaultdict, Counter
from pathlib import Path


def validate_dataset(tar_path):
    """Validate a single tar file and show statistics."""
    print(f"\n{'='*80}")
    print(f"Validating: {tar_path}")
    print(f"{'='*80}\n")

    if not Path(tar_path).exists():
        print(f"ERROR: File not found: {tar_path}")
        return

    # Decode helpers
    def decode_pth(data):
        return torch.load(io.BytesIO(data), weights_only=False)

    def decode_json(data):
        return json.loads(data.decode())

    # Load dataset
    dataset = (
        wds.WebDataset(tar_path)
        .decode(
            wds.handle_extension(".pth", decode_pth),
            wds.handle_extension(".json", decode_json),
        )
    )

    # Collect statistics
    timestep_counts = Counter()
    prompt_timesteps = defaultdict(set)  # Track which timesteps each prompt has
    total_samples = 0
    sample_types = Counter()
    shapes = defaultdict(set)

    print("Reading dataset...")
    for sample in dataset:
        total_samples += 1

        # Extract metadata
        metadata = sample.get('metadata.json', {})
        timestep_idx = metadata.get('timestep_idx', 'unknown')
        sample_type = metadata.get('type', 'unknown')
        prompt = metadata.get('prompt', 'unknown')[:50]  # Truncate for display

        timestep_counts[timestep_idx] += 1
        sample_types[sample_type] += 1

        # Track which timesteps this prompt has
        prompt_key = (sample_type, prompt)
        prompt_timesteps[prompt_key].add(timestep_idx)

        # Get shapes
        if 'input.pth' in sample:
            input_data = sample['input.pth']
            if isinstance(input_data, dict):
                hidden_states = input_data.get('hidden_states')
                if hidden_states is not None:
                    shapes['input_hidden_states'].add(tuple(hidden_states.shape))

        if 'target.pth' in sample:
            target = sample['target.pth']
            shapes['target'].add(tuple(target.shape))

    # Print statistics
    print(f"Total samples: {total_samples}")
    print(f"\nSample types:")
    for sample_type, count in sorted(sample_types.items()):
        print(f"  {sample_type}: {count} samples")

    print(f"\nTimestep distribution:")
    for timestep, count in sorted(timestep_counts.items()):
        print(f"  timestep {timestep}: {count} samples")

    print(f"\nShapes found:")
    for name, shape_set in sorted(shapes.items()):
        print(f"  {name}:")
        for shape in sorted(shape_set):
            print(f"    {shape}")

    # Analyze prompt coverage
    print(f"\nPrompt timestep coverage:")
    unique_prompts = len(set(prompt for _, prompt in prompt_timesteps.keys()))
    timesteps_per_prompt = [len(ts) for ts in prompt_timesteps.values()]

    if timesteps_per_prompt:
        min_ts = min(timesteps_per_prompt)
        max_ts = max(timesteps_per_prompt)
        avg_ts = sum(timesteps_per_prompt) / len(timesteps_per_prompt)

        print(f"  Unique prompts: {unique_prompts}")
        print(f"  Timesteps per prompt - min: {min_ts}, max: {max_ts}, avg: {avg_ts:.1f}")

        # Show example of which timesteps are present for first few prompts
        print(f"\nExample timestep indices (first 3 prompts):")
        for prompt_key, timesteps in list(prompt_timesteps.items())[:3]:
            sample_type, prompt = prompt_key
            sorted_timesteps = sorted(timesteps)
            print(f"  {sample_type[:6]} | {prompt[:40]:40s} | timesteps: {sorted_timesteps}")

    print(f"\n{'='*80}\n")


def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate timestep data in WebDataset tar files')
    parser.add_argument('tar_files', nargs='+', help='Path(s) to tar.gz files to validate')

    args = parser.parse_args()

    for tar_path in args.tar_files:
        validate_dataset(tar_path)


if __name__ == "__main__":
    main()
