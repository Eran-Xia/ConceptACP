"""
Automated Feature Selection for ConceptACP
Identifies SAE features that distinguish target concepts from retain concepts.
Supports both SDXL and SD 1.5 models.

Usage:
    # Select features for all unlearn types
    uv run src/conceptacp/utils/feature_select.py

    # Select features for specific unlearn type with SD 1.5
    uv run src/conceptacp/utils/feature_select.py data=sd15 unlearn_type=nudity

    # Select features for multiple blocks
    uv run src/conceptacp/utils/feature_select.py unlearn_type=nudity blocks='[down.2.1,mid.0]'
"""

import torch
import os
import json
import pathlib
from collections import defaultdict
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)


def generate_sparse_maps(cache, code, code_to_block, saes_dict, model_version, device, capture_interval=None):
    """
    Generate sparse activation maps from cached activations.

    Args:
        cache: Dictionary with 'input' and 'output' tensors
        code: Block code (e.g., 'down.2.1')
        code_to_block: Mapping from code to full block name
        saes_dict: Dictionary of loaded SAE models
        model_version: 'sdxl' or 'sd15'
        device: Device to run on
        capture_interval: Optional interval for timestep subsampling (e.g., 5 = every 5th timestep)

    Returns:
        sparse_maps: Tensor of shape [spatial_patches, num_features]
    """
    block = code_to_block[code]
    sae = saes_dict[code]

    # Get difference: output - input
    # Move to device for computation, then we'll move result back to CPU
    diff = cache["output"][block].to(device) - cache["input"][block].to(device)

    # Handle different tensor shapes between SDXL and SD 1.5
    # Treat each timestep as an independent data point (don't average!)
    # SDXL (hooked_sd_pipeline): [batch, timesteps, channels, height, width] (5D)
    # SD 1.5 (hooked_sd15_pipeline): [batch, timesteps, spatial_patches, features] (4D)

    if len(diff.shape) == 5:
        # SDXL: [batch, timesteps, channels, height, width] -> [batch*timesteps, H*W, C]
        batch, timesteps, channels, height, width = diff.shape

        # Apply capture_interval if specified to reduce memory
        if capture_interval is not None and timesteps > 1:
            timestep_indices = list(range(0, timesteps, capture_interval))
            diff = diff[:, timestep_indices, :, :, :]  # Select subset
            timesteps = len(timestep_indices)

        # Reshape to treat each timestep as a separate sample
        diff = diff.permute(0, 1, 3, 4, 2)  # [batch, timesteps, H, W, C]
        diff = diff.reshape(batch * timesteps, height * width, channels)  # [batch*timesteps, H*W, C]
    elif len(diff.shape) == 4:
        # SD 1.5: [batch, timesteps, spatial_patches, features] -> [batch*timesteps, spatial_patches, features]
        batch, timesteps, spatial_patches, features = diff.shape

        # Apply capture_interval if specified to reduce memory
        if capture_interval is not None and timesteps > 1:
            timestep_indices = list(range(0, timesteps, capture_interval))
            diff = diff[:, timestep_indices, :, :]  # Select subset
            timesteps = len(timestep_indices)

        # Reshape to treat each timestep as a separate sample
        diff = diff.reshape(batch * timesteps, spatial_patches, features)  # [batch*timesteps, spatial, features]
    else:
        raise ValueError(f"Unexpected diff shape: {diff.shape}")

    # Now diff is always 3D: [num_samples, spatial_patches, features]
    # where num_samples = batch * timesteps

    # Now diff is always 3D: [num_samples, spatial_patches, features]
    # where num_samples = batch * timesteps

    with torch.no_grad():
        if model_version == 'sdxl':
            # ConceptACP API: returns dense feature map
            # Process each sample separately and concatenate
            num_samples = diff.shape[0]
            sparse_maps_list = []
            for i in range(num_samples):
                sparse_maps_list.append(sae.encode(diff[i]))
            sparse_maps = torch.cat(sparse_maps_list, dim=0)
        elif model_version == 'sd15':
            # SAeUron API: expects [batch, seq_len, features], returns top-k
            # diff is already [num_samples, spatial, features] which is what we need
            num_samples, seq_len, features = diff.shape
            encoder_output = sae.encode(diff)  # Returns top-k for each sample

            # encoder_output.top_acts and top_indices: [num_samples, seq_len, k]
            k = encoder_output.top_acts.shape[-1]
            num_latents = sae.num_latents

            # Create dense representation from top-k for all samples
            # Final shape: [num_samples * seq_len, num_latents]
            sparse_maps = torch.zeros(num_samples * seq_len, num_latents,
                                     device=device, dtype=encoder_output.top_acts.dtype)

            # Reshape to flatten batch and spatial dimensions
            top_acts_flat = encoder_output.top_acts.reshape(-1, k)  # [num_samples*seq_len, k]
            top_indices_flat = encoder_output.top_indices.reshape(-1, k)  # [num_samples*seq_len, k]

            # Scatter using advanced indexing
            sample_indices = torch.arange(num_samples * seq_len, device=device)[:, None]
            sparse_maps[sample_indices, top_indices_flat] = top_acts_flat
        else:
            raise ValueError(f"Unknown model_version: {model_version}")

    # Free intermediate tensors
    del diff
    if model_version == 'sd15':
        del encoder_output, top_acts_flat, top_indices_flat, sample_indices

    return sparse_maps


def mu_score(sparse_maps):
    """Calculate mean activation across all spatial positions."""
    sparse_maps_flattened = sparse_maps.reshape(-1, sparse_maps.shape[-1])
    sparse_maps_mean = sparse_maps_flattened.mean(dim=0)
    return sparse_maps_mean


def score_features(forgot_maps, retain_maps):
    """
    Score features by comparing forgot vs retain activations.
    Higher score = feature activates more for forgot concepts.

    Returns:
        feature_score: Tensor of scores for each feature
    """
    mu_forgot = mu_score(forgot_maps)
    mu_retain = mu_score(retain_maps)

    # Normalized difference
    return mu_forgot / mu_forgot.sum(dim=-1) - mu_retain / mu_retain.sum(dim=-1)


def load_saes(cfg, device, dtype):
    """Load SAE models for all blocks."""
    logger.info(f"Loading SAE models for {cfg.model_version.upper()}...")

    saes_dict = {}
    means_dict = {}  # Only used for SDXL

    for code, block in cfg.code_to_block.items():
        if cfg.model_version == 'sdxl':
            from conceptacp.sae.SAE.sae import SparseAutoencoder
            sae = SparseAutoencoder.load_from_disk(
                os.path.join(cfg.sae_checkpoints, f"{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final"),
            )
            means = torch.load(
                os.path.join(cfg.sae_checkpoints, f"{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final", "mean.pt"),
                weights_only=True
            )
            saes_dict[code] = sae.to(device, dtype=dtype)
            means_dict[code] = means.to(device, dtype=dtype)
            logger.info(f"  Loaded SDXL SAE for {code} (d_model={sae.d_model})")

        elif cfg.model_version == 'sd15':
            from conceptacp.sae.SAE.sae_saeuron import Sae
            sae = Sae.load_from_disk(
                os.path.join(cfg.sae_checkpoints, block),
                device=device,
                decoder=True
            )
            sae = sae.to(dtype=dtype)
            saes_dict[code] = sae
            logger.info(f"  Loaded SD 1.5 SAE for {code} (d_in={sae.d_in}, num_latents={sae.num_latents})")
        else:
            raise ValueError(f"Unknown model_version: {cfg.model_version}")

    return saes_dict, means_dict


def load_pipeline(cfg, device, dtype):
    """Load hooked diffusion pipeline."""
    logger.info(f"Loading {cfg.model_version.upper()} pipeline from {cfg.model_path}...")

    if cfg.model_version == 'sdxl':
        from conceptacp.sae.SDLens import HookedStableDiffusionXLPipeline as HookedPipeline
    elif cfg.model_version == 'sd15':
        from conceptacp.sae.SDLens import HookedStableDiffusionPipeline as HookedPipeline
    else:
        raise ValueError(f"Unknown model_version: {cfg.model_version}")

    pipe = HookedPipeline.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map="balanced",
        variant=("fp16" if dtype == torch.float16 else None)
    )
    pipe.set_progress_bar_config(disable=True)

    return pipe


def collect_activations(pipe, prompts, unlearn_type, code_to_block, saes_dict,
                       model_version, device, num_inference_steps=1, guidance_scale=0.0, capture_interval=None, desc="Processing"):
    """Collect SAE activations for a list of prompts."""
    activation_maps = defaultdict(list)

    for img_class in prompts[unlearn_type]:
        for prompt in tqdm(prompts[unlearn_type][img_class], desc=f"{desc} {img_class}"):
            output, cache = pipe.run_with_cache(
                prompt,
                positions_to_cache=list(code_to_block.values()),
                save_input=True,
                save_output=True,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(42)
            )

            for code in code_to_block:
                sparse_maps = generate_sparse_maps(
                    cache, code, code_to_block, saes_dict, model_version, device, capture_interval
                )
                # Immediately move to CPU to free GPU memory
                activation_maps[code].append(sparse_maps.cpu())

            # Clear cache to free GPU memory for next prompt
            del cache
            torch.cuda.empty_cache()

    # Stack into tensors (keep on CPU to save GPU memory)
    activation_maps = {k: torch.stack(v) for k, v in activation_maps.items()}

    # Log info about data collected
    if activation_maps:
        sample_code = list(activation_maps.keys())[0]
        sample_maps = activation_maps[sample_code]
        num_prompts = len(activation_maps[sample_code])
        if capture_interval is not None and num_inference_steps > 1:
            actual_timesteps = len(list(range(0, num_inference_steps, capture_interval)))
            logger.info(f"  → Collected {num_prompts} prompts × {actual_timesteps} timesteps (interval={capture_interval}, stored on CPU)")
        else:
            logger.info(f"  → Collected {num_prompts} prompts × {num_inference_steps} timesteps each (stored on CPU)")

    return activation_maps


def select_features_for_unlearn_type(cfg, pipe, saes_dict, retain_prompts,
                                     forgot_prompts, unlearn_type, device):
    """
    Select features for a specific unlearn type.

    Returns:
        result: Dictionary with unlearn_type, block, finetune_feature, finetune_strength
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing unlearn_type: {unlearn_type}")
    logger.info(f"{'='*60}")

    # Collect activations
    logger.info("Collecting retain activations...")
    retain_maps = collect_activations(
        pipe, retain_prompts, unlearn_type, cfg.code_to_block,
        saes_dict, cfg.model_version, device,
        num_inference_steps=cfg.get('num_inference_steps', 1),
        guidance_scale=cfg.guidance_scale,
        capture_interval=cfg.get('capture_interval', None),
        desc="Retain"
    )

    logger.info("Collecting forgot activations...")
    forgot_maps = collect_activations(
        pipe, forgot_prompts, unlearn_type, cfg.code_to_block,
        saes_dict, cfg.model_version, device,
        num_inference_steps=cfg.get('num_inference_steps', 1),
        guidance_scale=cfg.guidance_scale,
        capture_interval=cfg.get('capture_interval', None),
        desc="Forgot"
    )

    # Score features for each block
    logger.info("\nScoring features across blocks...")
    block_top_scores = {}
    block_features = {}

    for code in cfg.code_to_block:
        feature_score = score_features(forgot_maps[code], retain_maps[code])
        sorted_indices = torch.argsort(feature_score, descending=True)

        top_k = cfg.get('display_top_k', 10)
        top_score = feature_score[sorted_indices[0]].item()
        block_top_scores[code] = top_score
        block_features[code] = (feature_score, sorted_indices)

        logger.info(f"\n  Block: {code}")
        logger.info(f"    Top score: {top_score:.6f}")
        logger.info(f"    Top {top_k} indices: {sorted_indices[:top_k].tolist()}")
        logger.info(f"    Top {top_k} scores: {feature_score[sorted_indices[:top_k]].tolist()}")

    # Select best block
    target_block = max(block_top_scores, key=block_top_scores.get)
    logger.info(f"\n✓ Selected block: {target_block} (score: {block_top_scores[target_block]:.6f})")

    # Select features from best block
    feature_score, sorted_indices = block_features[target_block]
    max_score = feature_score[sorted_indices[0]].item()

    # Threshold: features with score >= max_score / threshold_divisor
    threshold_divisor = cfg.get('threshold_divisor', 5)
    threshold = max_score / threshold_divisor

    selected_features = []
    selected_strengths = []

    for idx in sorted_indices:
        score_value = feature_score[idx].item()
        if score_value >= threshold:
            # Scale strength proportionally to score
            if score_value == max_score:
                strength = cfg.get('max_strength', -5.0)
            else:
                strength = cfg.get('max_strength', -5.0) * (score_value / max_score)

            strength = round(strength, 1)
            selected_features.append(idx.item())
            selected_strengths.append(strength)
        else:
            break

    logger.info(f"\n✓ Selected {len(selected_features)} features (threshold: {threshold:.6f})")
    logger.info(f"  Features: {selected_features}")
    logger.info(f"  Strengths: {selected_strengths}")

    result = {
        "unlearn_type": unlearn_type,
        "block": [target_block],
        "finetune_feature": selected_features,
        "finetune_strength": selected_strengths
    }

    return result


@hydra.main(version_base=None, config_path="conf", config_name="feature_select")
def main(cfg: DictConfig):
    """Main feature selection script."""

    # Setup
    device = torch.device(cfg.get('device', 'cuda:0'))
    dtype = torch.float32

    logger.info("="*80)
    logger.info("ConceptACP Feature Selection")
    logger.info("="*80)
    logger.info(f"Model version: {cfg.model_version}")
    logger.info(f"Model path: {cfg.model_path}")
    logger.info(f"SAE checkpoints: {cfg.sae_checkpoints}")
    logger.info(f"Device: {device}")
    logger.info(f"Blocks to analyze: {list(cfg.code_to_block.keys())}")

    # Load pipeline and SAEs
    pipe = load_pipeline(cfg, device, dtype)
    saes_dict, means_dict = load_saes(cfg, device, dtype)

    # Load prompts
    logger.info("\nLoading prompts...")

    # Check if this is an UnlearnCanvas task (starts with 'uc_')
    is_unlearncanvas = cfg.get('unlearn_type', '').startswith('uc_')

    if is_unlearncanvas:
        # For UnlearnCanvas: use retain_featureselect.json for feature selection
        logger.info("UnlearnCanvas task detected - using retain_featureselect.json")
        retain_featureselect_file = pathlib.Path(cfg.work_dir) / "data/train_prompts/retain_featureselect.json"
        if retain_featureselect_file.exists():
            with open(retain_featureselect_file, 'r') as f:
                retain_prompts = json.load(f)
            logger.info(f"  Loaded retain prompts from {retain_featureselect_file}")
        else:
            logger.warning(f"retain_featureselect.json not found at {retain_featureselect_file}, falling back to retain.json")
            with open(cfg.retain_prompts, 'r') as f:
                retain_prompts = json.load(f)
    else:
        with open(cfg.retain_prompts, 'r') as f:
            retain_prompts = json.load(f)

    with open(cfg.forgot_prompts, 'r') as f:
        forgot_prompts = json.load(f)

    # Determine which unlearn types to process
    if cfg.get('unlearn_type') and cfg.unlearn_type != 'all':
        unlearn_types = [cfg.unlearn_type]
    else:
        # Process all unlearn types found in prompts
        unlearn_types = list(set(retain_prompts.keys()) & set(forgot_prompts.keys()))

    logger.info(f"Unlearn types to process: {unlearn_types}")

    # Process each unlearn type
    results = []
    for unlearn_type in unlearn_types:
        result = select_features_for_unlearn_type(
            cfg, pipe, saes_dict, retain_prompts, forgot_prompts,
            unlearn_type, device
        )
        results.append(result)

        # Save config file
        output_dir = pathlib.Path(cfg.finetune_config_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{cfg.model_version}_{unlearn_type}.json"

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)

        logger.info(f"\n✓ Config saved to {output_path}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Feature Selection Complete!")
    logger.info("="*80)
    for result in results:
        logger.info(f"{result['unlearn_type']:20s} → {result['block'][0]:10s} ({len(result['finetune_feature'])} features)")

    logger.info("\nNext steps:")
    logger.info("  1. Review generated configs in: " + str(cfg.finetune_config_dir))
    logger.info("  2. Generate training data: uv run src/conceptacp/utils/make_block_data.py")
    logger.info("  3. Train model: uv run src/conceptacp/utils/finetune_block.py")


if __name__ == "__main__":
    main()
