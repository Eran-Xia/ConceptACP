"""
Efficient data generation for block-level fine-tuning.
Captures (input, target_output) pairs for a specific attention block.
"""
from conceptacp.sae.SDLens import HookedStableDiffusionXLPipeline
from conceptacp.sae.SAE import SparseAutoencoder
import torch
import os
import hydra
from omegaconf import DictConfig
import pathlib
import json
import webdataset as wds


def scale_features(sparse_maps, feature_scales):
    """Scale specific SAE features according to the config."""
    for idx, scale in feature_scales.items():
        sparse_maps[:, :, idx] = sparse_maps[:, :, idx] * scale
    return sparse_maps


class BlockInputCapturer:
    """Captures inputs and outputs for a specific attention block."""

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def capture_hook(self, module, input_tuple, kwargs, output):
        """Hook that captures both inputs and outputs of the attention block."""
        # Store the input args and kwargs
        # For attention blocks: input[0] is hidden_states
        input_data = {
            'hidden_states': input_tuple[0].detach().cpu(),
            'encoder_hidden_states': kwargs.get('encoder_hidden_states', None),
            'attention_mask': kwargs.get('attention_mask', None),
        }

        # Handle encoder_hidden_states which might be a tensor
        if input_data['encoder_hidden_states'] is not None:
            input_data['encoder_hidden_states'] = input_data['encoder_hidden_states'].detach().cpu()

        # Handle attention_mask
        if input_data['attention_mask'] is not None:
            input_data['attention_mask'] = input_data['attention_mask'].detach().cpu()

        # Handle output (might be tuple or tensor)
        if isinstance(output, tuple):
            output_tensor = output[0].detach().cpu()
        else:
            output_tensor = output.detach().cpu()

        self.inputs.append(input_data)
        self.outputs.append(output_tensor)

    def reset(self):
        """Clear captured data."""
        self.inputs = []
        self.outputs = []


@hydra.main(config_path="conf/data", config_name="default")
def main(cfg: DictConfig):
    device = torch.device("cuda:0")
    dtype = torch.float32

    # Load pipeline
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map="balanced",
        variant=("fp16" if dtype == torch.float16 else None)
    )
    pipe.set_progress_bar_config(disable=True)

    # Load SAE for the target block
    path_to_checkpoints = cfg.sae_checkpoints

    # Load finetune config to get the target block and features
    finetune_config_dir = cfg.finetune_config_dir
    finetune_config_file = pathlib.Path(finetune_config_dir) / (cfg.unlearn_type + ".json")
    with open(finetune_config_file, 'r') as f:
        finetune_config = json.load(f)
        feature_scales = {}
        for feature, scale in zip(finetune_config["finetune_feature"], finetune_config["finetune_strength"]):
            feature_scales[feature] = scale * cfg.make_data_scale
        finetune_block = finetune_config["block"][0]
        finetune_type = finetune_config["unlearn_type"]

    # Load SAE for the target block
    block_name = cfg.code_to_block[finetune_block]
    sae = SparseAutoencoder.load_from_disk(
        os.path.join(path_to_checkpoints, f"{block_name}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final"),
    )
    means = torch.load(
        os.path.join(path_to_checkpoints, f"{block_name}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final", "mean.pt"),
        weights_only=True
    )
    sae = sae.to('cuda', dtype=dtype)
    means = means.to('cuda', dtype=dtype)

    # Load prompts
    with open(cfg.retain_prompts, 'r') as f:
        retain = json.load(f)
    with open(cfg.forgot_prompts, 'r') as f:
        forgot = json.load(f)

    forgot_prompts = []
    for img_class in forgot[cfg.unlearn_type]:
        for prompt in forgot[cfg.unlearn_type][img_class]:
            forgot_prompts.append(prompt)

    retain_prompts = []
    for img_class in retain[cfg.unlearn_type]:
        for prompt in retain[cfg.unlearn_type][img_class]:
            retain_prompts.append(prompt)

    # Create output directory
    recons_save_path = cfg.recons_save_path
    os.makedirs(recons_save_path, exist_ok=True)

    # Determine which timesteps to capture
    capture_timesteps = cfg.get('capture_timesteps', None)
    if capture_timesteps is None or capture_timesteps == 'all':
        capture_all = True
        timesteps_to_capture = None
    else:
        capture_all = False
        timesteps_to_capture = set(capture_timesteps)

    # Build filename suffix with timestep information
    timestep_suffix = f"_steps{cfg.num_inference_steps}"
    if not capture_all:
        # Format: _steps10_t0-5-9 for selective capture
        timestep_list = '-'.join(map(str, sorted(timesteps_to_capture)))
        timestep_suffix += f"_t{timestep_list}"

    print(f"Processing {len(forgot_prompts)} forgot prompts and {len(retain_prompts)} retain prompts")
    print(f"Target block: {finetune_block} -> {block_name}")
    print(f"Feature scales: {feature_scales}")
    print(f"Num inference steps: {cfg.num_inference_steps}")
    print(f"Capture timesteps: {'all' if capture_all else list(timesteps_to_capture)}")
    print(f"Filename suffix: {timestep_suffix}")

    # ===== Process FORGOT prompts (with SAE modification) =====
    print("\n=== Processing FORGOT prompts ===")
    capturer = BlockInputCapturer()

    forgot_filename = f'{recons_save_path}/{finetune_block}_{finetune_type}_{cfg.make_data_scale}_block{timestep_suffix}.tar.gz'
    writer = wds.TarWriter(forgot_filename)

    # Register hook to capture inputs at the target block
    target_module = pipe._locate_block(block_name)
    hook_handle = target_module.register_forward_hook(capturer.capture_hook, with_kwargs=True)

    for idx, prompt in enumerate(forgot_prompts):
        capturer.reset()

        # Run with cache to get input/output
        _, cache = pipe.run_with_cache(
            prompt,
            positions_to_cache=[block_name],
            save_input=True,
            save_output=True,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=0.0,
            generator=torch.Generator(device="cpu").manual_seed(42)
        )

        # Get the difference (output - input) at the target block
        block_input = cache["input"][block_name].to(device)
        block_output = cache["output"][block_name].to(device)

        # Process each timestep separately through the SAE
        # Shape: [batch, timesteps, channels, height, width]
        batch_size, num_timesteps, channels, height, width = block_input.shape

        # Initialize target output with same shape as block_input
        target_output = torch.zeros_like(block_input)

        with torch.no_grad():
            for t_idx in range(num_timesteps):
                # Extract single timestep
                input_t = block_input[:, t_idx:t_idx+1]  # [batch, 1, C, H, W]
                output_t = block_output[:, t_idx:t_idx+1]  # [batch, 1, C, H, W]
                diff_t = output_t - input_t

                # Reshape for SAE: [batch, 1, channels, height, width] -> [seq_len, features]
                diff_reshaped = diff_t.permute(0, 1, 3, 4, 2).squeeze(0).squeeze(0)  # [H*W, C]

                # Encode with SAE, scale features, decode
                sparse_maps = sae.encode(diff_reshaped)
                sparse_maps = scale_features(sparse_maps, feature_scales)
                modified_recons = sae.decoder(sparse_maps) + sae.pre_bias

                # Reshape back and add to input to get target for this timestep
                # modified_recons: [H*W, C] -> [1, 1, C, H, W]
                modified_recons_reshaped = modified_recons.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3)
                target_output[:, t_idx:t_idx+1] = input_t + modified_recons_reshaped

        # Now we have all the captured inputs from the hook
        # For forgot prompts, we need to save timestep inputs with the modified output
        num_timesteps = len(capturer.inputs)

        for t_idx in range(num_timesteps):
            # Skip this timestep if not in the capture list
            if not capture_all and t_idx not in timesteps_to_capture:
                continue

            input_data = capturer.inputs[t_idx]

            # Save each timestep as a separate sample
            writer.write({
                "__key__": f"forgot_{idx}_t{t_idx}",
                "input.pth": {
                    'hidden_states': input_data['hidden_states'].half(),
                    'encoder_hidden_states': input_data['encoder_hidden_states'].half() if input_data['encoder_hidden_states'] is not None else None,
                    'attention_mask': input_data['attention_mask'].half() if input_data['attention_mask'] is not None else None,
                },
                "target.pth": target_output[:, t_idx:t_idx+1].half().cpu(),  # Single timestep
                "metadata.json": {
                    'prompt': prompt,
                    'timestep_idx': t_idx,
                    'type': 'forgot',
                    'total_timesteps': num_timesteps
                }
            })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(forgot_prompts)} forgot prompts")

    hook_handle.remove()
    writer.close()
    print(f"Saved forgot data to {forgot_filename}")

    # ===== Process RETAIN prompts (without SAE modification) =====
    print("\n=== Processing RETAIN prompts ===")
    capturer.reset()

    retain_filename = f'{recons_save_path}/{finetune_block}_{finetune_type}_retain_block{timestep_suffix}.tar.gz'
    writer = wds.TarWriter(retain_filename)

    hook_handle = target_module.register_forward_hook(capturer.capture_hook, with_kwargs=True)

    for idx, prompt in enumerate(retain_prompts):
        capturer.reset()

        # Run inference
        _, cache = pipe.run_with_cache(
            prompt,
            positions_to_cache=[block_name],
            save_input=True,
            save_output=True,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=0.0,
            generator=torch.Generator(device="cpu").manual_seed(42)
        )

        # For retain prompts, target is the original output (no modification)
        target_output = cache["output"][block_name].to(device)

        num_timesteps = len(capturer.inputs)

        for t_idx in range(num_timesteps):
            # Skip this timestep if not in the capture list
            if not capture_all and t_idx not in timesteps_to_capture:
                continue

            input_data = capturer.inputs[t_idx]

            writer.write({
                "__key__": f"retain_{idx}_t{t_idx}",
                "input.pth": {
                    'hidden_states': input_data['hidden_states'].half(),
                    'encoder_hidden_states': input_data['encoder_hidden_states'].half() if input_data['encoder_hidden_states'] is not None else None,
                    'attention_mask': input_data['attention_mask'].half() if input_data['attention_mask'] is not None else None,
                },
                "target.pth": target_output[:, t_idx:t_idx+1].half().cpu(),
                "metadata.json": {
                    'prompt': prompt,
                    'timestep_idx': t_idx,
                    'type': 'retain',
                    'total_timesteps': num_timesteps
                }
            })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(retain_prompts)} retain prompts")

    hook_handle.remove()
    writer.close()
    print(f"Saved retain data to {retain_filename}")

    print("\n=== Data generation complete ===")
    print(f"\nSummary:")
    print(f"  Forgot prompts: {len(forgot_prompts)}")
    print(f"  Retain prompts: {len(retain_prompts)}")
    print(f"  Inference steps: {cfg.num_inference_steps}")
    if capture_all:
        print(f"  Timesteps captured: all ({cfg.num_inference_steps} per prompt)")
        print(f"  Expected total samples: {len(forgot_prompts) * cfg.num_inference_steps} forgot + {len(retain_prompts) * cfg.num_inference_steps} retain")
    else:
        print(f"  Timesteps captured: {sorted(timesteps_to_capture)}")
        print(f"  Expected total samples: {len(forgot_prompts) * len(timesteps_to_capture)} forgot + {len(retain_prompts) * len(timesteps_to_capture)} retain")
    print(f"\nFiles saved:")
    print(f"  Forgot: {forgot_filename}")
    print(f"  Retain: {retain_filename}")
    print(f"\nTo validate the data, run:")
    print(f"  uv run src/conceptacp/utils/validate_timesteps.py \\")
    print(f"    {forgot_filename} \\")
    print(f"    {retain_filename}")


if __name__ == "__main__":
    main()
