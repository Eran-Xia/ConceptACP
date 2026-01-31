"""
Efficient data generation for block-level fine-tuning.
Captures (input, target_output) pairs for a specific attention block.
Supports both SD 1.5 and SDXL models.
"""
import torch
import os
import hydra
from omegaconf import DictConfig
import pathlib
import json
import webdataset as wds


def scale_features(sparse_maps, feature_scales):
    """
    Scale specific SAE features according to the config.

    Args:
        sparse_maps: Tensor of shape [seq_len, n_dirs] containing SAE activations
        feature_scales: Dict mapping feature indices to scale factors

    Returns:
        scaled_sparse_maps: Tensor with scaled features
    """
    for idx, scale in feature_scales.items():
        sparse_maps[:, idx] = sparse_maps[:, idx] * scale
    return sparse_maps


def scale_features_topk(top_acts, top_indices, feature_scales):
    """
    Scale specific SAE features for top-k representation (SAeUron format).

    Args:
        top_acts: Tensor of shape [seq_len, k] containing top-k activations
        top_indices: Tensor of shape [seq_len, k] containing top-k feature indices
        feature_scales: Dict mapping feature indices to scale factors

    Returns:
        scaled_top_acts: Tensor with scaled activations
    """
    scaled_top_acts = top_acts.clone()
    for feat_idx, scale in feature_scales.items():
        # Find where this feature appears in the top-k
        mask = (top_indices == feat_idx)
        scaled_top_acts[mask] = scaled_top_acts[mask] * scale
    return scaled_top_acts


def encode_and_scale_sae(sae, diff_reshaped, feature_scales, model_version):
    """
    Encode with SAE, scale features, and decode.
    Handles both ConceptACP (SDXL) and SAeUron (SD15) SAE APIs.

    Args:
        sae: SAE model
        diff_reshaped: Residual to encode [seq_len, d_in]
        feature_scales: Dict of feature indices to scale factors
        model_version: 'sdxl' or 'sd15'

    Returns:
        modified_recons: Scaled reconstruction [seq_len, d_in]
    """
    if model_version == 'sdxl':
        # ConceptACP API: dense feature maps
        sparse_maps = sae.encode(diff_reshaped)
        sparse_maps = scale_features(sparse_maps, feature_scales)
        modified_recons = sae.decoder(sparse_maps) + sae.pre_bias
    elif model_version == 'sd15':
        # SAeUron API: top-k indices and activations
        # Input: diff_reshaped [seq_len, d_in]
        # After unsqueeze: [1, seq_len, d_in]
        # SAE encode internally flattens to [1*seq_len, d_in] = [seq_len, d_in]
        # Returns: EncoderOutput(top_acts=[seq_len, k], top_indices=[seq_len, k])
        encoder_output = sae.encode(diff_reshaped.unsqueeze(0))

        # Scale features - output is already [seq_len, k], no squeeze needed
        scaled_top_acts = scale_features_topk(
            encoder_output.top_acts,  # [seq_len, k]
            encoder_output.top_indices,  # [seq_len, k]
            feature_scales
        )

        # Decode - takes [seq_len, k], returns [seq_len, d_in]
        modified_recons = sae.decode(
            scaled_top_acts,  # [seq_len, k]
            encoder_output.top_indices  # [seq_len, k]
        )  # Returns [seq_len, d_in]

        # Validate output shape matches input
        assert modified_recons.shape == diff_reshaped.shape, \
            f"SAE reconstruction shape mismatch: expected {diff_reshaped.shape}, got {modified_recons.shape}"
    else:
        raise ValueError(f"Unknown model_version: {model_version}")

    return modified_recons


class BlockInputCapturer:
    """Captures inputs and outputs for a specific attention block."""

    def __init__(self, guidance_scale=0.0):
        self.inputs = []
        self.outputs = []
        self.guidance_scale = guidance_scale

    def _extract_conditional(self, tensor):
        """Extract conditional part when using classifier-free guidance.

        When guidance_scale > 1.0, batch contains [unconditional, conditional].
        We only want the conditional part for training.
        """
        if tensor is None:
            return None

        if self.guidance_scale > 1.0:
            batch_size = tensor.shape[0]
            if batch_size % 2 == 0 and batch_size >= 2:
                # Split doubled batch: [uncond, cond] -> cond
                _, cond = tensor.chunk(2)
                return cond
        # No CFG or already single batch
        return tensor

    def capture_hook(self, module, input_tuple, kwargs, output):
        """Hook that captures both inputs and outputs of the attention block."""
        # Store the input args and kwargs
        # For attention blocks: input[0] is hidden_states
        hidden_states = input_tuple[0].detach()
        encoder_hidden_states = kwargs.get('encoder_hidden_states', None)
        attention_mask = kwargs.get('attention_mask', None)

        # Extract conditional parts if using CFG
        hidden_states = self._extract_conditional(hidden_states)
        encoder_hidden_states = self._extract_conditional(encoder_hidden_states) if encoder_hidden_states is not None else None
        attention_mask = self._extract_conditional(attention_mask) if attention_mask is not None else None

        input_data = {
            'hidden_states': hidden_states.cpu(),
            'encoder_hidden_states': encoder_hidden_states.cpu() if encoder_hidden_states is not None else None,
            'attention_mask': attention_mask.cpu() if attention_mask is not None else None,
        }

        # Handle output (might be tuple or tensor)
        if isinstance(output, tuple):
            output_tensor = output[0].detach()
        else:
            output_tensor = output.detach()

        # Extract conditional part of output
        output_tensor = self._extract_conditional(output_tensor).cpu()

        self.inputs.append(input_data)
        self.outputs.append(output_tensor)

    def reset(self):
        """Clear captured data."""
        self.inputs = []
        self.outputs = []


@hydra.main(version_base=None, config_path="conf", config_name="make_block_data")
def main(cfg: DictConfig):
    device = torch.device("cuda:0")
    dtype = torch.float32

    # Conditional pipeline loading based on model version
    if cfg.model_version == 'sdxl':
        from conceptacp.sae.SDLens import HookedStableDiffusionXLPipeline as HookedPipeline
        print("Loading SDXL pipeline")
    elif cfg.model_version == 'sd15':
        from conceptacp.sae.SDLens import HookedStableDiffusionPipeline as HookedPipeline
        print("Loading SD 1.5 pipeline")
    else:
        raise ValueError(f"Unknown model_version: {cfg.model_version}. Must be 'sdxl' or 'sd15'")

    # Load pipeline
    pipe = HookedPipeline.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map="balanced",
        variant=("fp16" if dtype == torch.float16 else None)
    )
    pipe.set_progress_bar_config(disable=True)

    # Load SAE for the target block
    path_to_checkpoints = cfg.sae_checkpoints

    # Load finetune config to get the target block and features
    # Check for model-prefixed file first (new pattern), then fall back to old patterns
    finetune_config_dir = pathlib.Path(cfg.finetune_config_dir)
    model_prefixed_file = finetune_config_dir / f"{cfg.model_version}_{cfg.unlearn_type}.json"
    model_specific_file = finetune_config_dir / cfg.model_version / f"{cfg.unlearn_type}.json"
    model_agnostic_file = finetune_config_dir / f"{cfg.unlearn_type}.json"

    if model_prefixed_file.exists():
        finetune_config_file = model_prefixed_file
        print(f"Using model-prefixed config: {finetune_config_file}")
    elif model_specific_file.exists():
        finetune_config_file = model_specific_file
        print(f"Using model-specific config (legacy): {finetune_config_file}")
    elif model_agnostic_file.exists():
        finetune_config_file = model_agnostic_file
        print(f"Using model-agnostic config: {finetune_config_file}")
    else:
        raise FileNotFoundError(f"No finetune config found for {cfg.unlearn_type} in {finetune_config_dir}")

    with open(finetune_config_file, 'r') as f:
        finetune_config = json.load(f)
        feature_scales = {}
        for feature, scale in zip(finetune_config["finetune_feature"], finetune_config["finetune_strength"]):
            feature_scales[feature] = scale * cfg.make_data_scale
        finetune_block = finetune_config["block"][0]
        finetune_type = finetune_config["unlearn_type"]

    # Load SAE for the target block
    block_name = cfg.code_to_block[finetune_block]

    # Conditional SAE loading based on model version
    if cfg.model_version == 'sdxl':
        from conceptacp.sae.SAE.sae import SparseAutoencoder
        sae = SparseAutoencoder.load_from_disk(
            os.path.join(path_to_checkpoints, f"{block_name}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final"),
        )
        means = torch.load(
            os.path.join(path_to_checkpoints, f"{block_name}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final", "mean.pt"),
            weights_only=True
        )
        sae = sae.to('cuda', dtype=dtype)
        means = means.to('cuda', dtype=dtype)
        print(f"Loaded SDXL SAE with d_model={sae.d_model}")
    elif cfg.model_version == 'sd15':
        from conceptacp.sae.SAE.sae_saeuron import Sae
        # SAeUron checkpoint structure: block_name/cfg.json and sae.safetensors
        # Load to CPU first to avoid OOM, then move to GPU
        print("Loading SAE to CPU first...")
        torch.cuda.empty_cache()  # Clear GPU cache before loading
        sae = Sae.load_from_disk(
            os.path.join(path_to_checkpoints, block_name),
            device='cpu',
            decoder=True
        )
        print(f"Moving SAE to GPU...")
        sae = sae.to('cuda', dtype=dtype)
        # SAeUron SAEs don't use separate means file - they have built-in bias
        means = None
        print(f"Loaded SD 1.5 SAE with d_in={sae.d_in}")

        # Validate dimensions match config
        assert sae.d_in == cfg.d_model, \
            f"SAE d_in {sae.d_in} doesn't match config d_model {cfg.d_model}"
    else:
        raise ValueError(f"Unknown model_version: {cfg.model_version}")

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
    capture_interval = cfg.get('capture_interval', None)
    capture_timesteps = cfg.get('capture_timesteps', None)

    if capture_interval is not None:
        # Method 1: Capture at regular intervals (takes precedence)
        capture_all = False
        # Generate timestep indices: 0, interval, 2*interval, ...
        timesteps_to_capture = set(range(0, cfg.num_inference_steps, capture_interval))
        print(f"Capturing every {capture_interval} steps: {sorted(timesteps_to_capture)}")
    elif capture_timesteps is None or capture_timesteps == 'all':
        # Method 2: Capture all timesteps
        capture_all = True
        timesteps_to_capture = None
    else:
        # Method 3: Capture specific timesteps
        capture_all = False
        timesteps_to_capture = set(capture_timesteps)

    # Build filename suffix with timestep information
    timestep_suffix = f"_steps{cfg.num_inference_steps}"
    if not capture_all:
        if capture_interval is not None:
            # Format: _steps50_interval5 for interval-based capture
            timestep_suffix += f"_interval{capture_interval}"
        else:
            # Format: _steps10_t0-5-9 for selective capture
            timestep_list = '-'.join(map(str, sorted(timesteps_to_capture)))
            timestep_suffix += f"_t{timestep_list}"

    print(f"Processing {len(forgot_prompts)} forgot prompts and {len(retain_prompts)} retain prompts")
    print(f"Target block: {finetune_block} -> {block_name}")
    print(f"Feature scales: {feature_scales}")
    print(f"Num inference steps: {cfg.num_inference_steps}")
    if capture_all:
        print(f"Capture mode: all timesteps")
    elif capture_interval is not None:
        print(f"Capture mode: every {capture_interval} steps ({len(timesteps_to_capture)} timesteps)")
    else:
        print(f"Capture mode: specific timesteps {sorted(timesteps_to_capture)}")
    print(f"Filename suffix: {timestep_suffix}")

    # ===== Process FORGOT prompts (with SAE modification) =====
    print("\n=== Processing FORGOT prompts ===")
    capturer = BlockInputCapturer(guidance_scale=cfg.guidance_scale)

    # Create output directory
    os.makedirs(recons_save_path, exist_ok=True)

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
            guidance_scale=cfg.guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(42)
        )

        # Get the difference (output - input) at the target block
        block_input = cache["input"][block_name].to(device)
        block_output = cache["output"][block_name].to(device)

        # Handle different formats for SDXL vs SD 1.5
        if len(block_input.shape) == 5:
            # SDXL format: [batch, timesteps, channels, height, width]
            batch_size, num_timesteps, channels, height, width = block_input.shape
            is_sequence_format = False
        elif len(block_input.shape) == 4:
            # SD 1.5 format: [batch, timesteps, seq_len, features]
            batch_size, num_timesteps, seq_len, features = block_input.shape
            is_sequence_format = True
        else:
            raise ValueError(f"Unexpected block_input shape: {block_input.shape}")

        # Initialize target output with same shape as block_input
        target_output = torch.zeros_like(block_input)

        with torch.no_grad():
            for t_idx in range(num_timesteps):
                if is_sequence_format:
                    # SD 1.5: Already in [batch, timesteps, seq_len, features] format
                    input_t = block_input[:, t_idx:t_idx+1]  # [batch, 1, seq_len, features]
                    output_t = block_output[:, t_idx:t_idx+1]  # [batch, 1, seq_len, features]
                    diff_t = output_t - input_t

                    # Reshape for SAE: [batch, 1, seq_len, features] -> [seq_len, features]
                    diff_reshaped = diff_t.squeeze(0).squeeze(0)  # [seq_len, features]

                    # Encode with SAE, scale features, decode
                    modified_recons = encode_and_scale_sae(
                        sae, diff_reshaped, feature_scales, cfg.model_version
                    )

                    # Reshape back: [seq_len, features] -> [1, 1, seq_len, features]
                    modified_recons_reshaped = modified_recons.unsqueeze(0).unsqueeze(0)
                    target_output[:, t_idx:t_idx+1] = input_t + modified_recons_reshaped
                else:
                    # SDXL: Spatial format [batch, timesteps, channels, height, width]
                    input_t = block_input[:, t_idx:t_idx+1]  # [batch, 1, C, H, W]
                    output_t = block_output[:, t_idx:t_idx+1]  # [batch, 1, C, H, W]
                    diff_t = output_t - input_t

                    # Reshape for SAE: [batch, 1, channels, height, width] -> [seq_len, features]
                    diff_reshaped = diff_t.permute(0, 1, 3, 4, 2).squeeze(0).squeeze(0)  # [H*W, C]

                    # Encode with SAE, scale features, decode
                    modified_recons = encode_and_scale_sae(
                        sae, diff_reshaped, feature_scales, cfg.model_version
                    )

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
                    'total_timesteps': num_timesteps,
                    'block_code': finetune_block,  # Short code (e.g., "up.1.2")
                    'block_name': block_name  # Full name (e.g., "up_blocks.1.attentions.2")
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
            guidance_scale=cfg.guidance_scale,
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
                    'total_timesteps': num_timesteps,
                    'block_code': finetune_block,  # Short code (e.g., "up.1.2")
                    'block_name': block_name  # Full name (e.g., "up_blocks.1.attentions.2")
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
