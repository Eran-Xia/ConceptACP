import json
import pathlib
import torch
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig
# from datasets import load_dataset

device = torch.device("cuda:1")


def load_unlearncanvas_constants(resources_path):
    """Load UnlearnCanvas style and object lists from const.py"""
    const_path = pathlib.Path(resources_path) / "const.py"
    if not const_path.exists():
        raise FileNotFoundError(f"const.py not found at {const_path}")

    # Add the resources path to Python path temporarily
    sys.path.insert(0, str(resources_path))
    try:
        from const import theme_available, class_available
        return theme_available, class_available
    finally:
        sys.path.pop(0)


def generate_unlearncanvas_test_prompts(unlearn_type, target_name, resources_path):
    """Generate test prompts for UnlearnCanvas tasks dynamically.

    Args:
        unlearn_type: e.g., 'uc_style_vangogh' or 'uc_object_bears'
        target_name: e.g., 'Van_Gogh' or 'Bears'
        resources_path: Path to UnlearnCanvas_resources

    Returns:
        forgot_prompts: Dict with prompts for the target concept
        retain_prompts: Dict with prompts for other concepts
    """
    styles, objects = load_unlearncanvas_constants(resources_path)

    forgot_prompts = {}
    retain_prompts = {}

    if unlearn_type.startswith('uc_style_'):
        # Style erasure: forgot = all objects in target style, retain = target style excluded
        forgot_prompts[unlearn_type] = {
            "style": [f"An image of {obj} in {target_name} style" for obj in objects]
        }
        # Retain: same objects but in other styles (sample a few)
        other_styles = [s for s in styles if s != target_name][:5]  # Use 5 other styles
        retain_prompts[unlearn_type] = {}
        for style in other_styles:
            retain_prompts[unlearn_type][style] = [
                f"An image of {obj} in {style} style" for obj in objects[:5]  # 5 objects per style
            ]
    else:  # uc_object_*
        # Object erasure: forgot = target object in all styles, retain = other objects
        forgot_prompts[unlearn_type] = {
            "object": [f"An image of {target_name} in {style} style" for style in styles]
        }
        # Retain: other objects in various styles
        other_objects = [o for o in objects if o != target_name]
        retain_prompts[unlearn_type] = {}
        for obj in other_objects:
            retain_prompts[unlearn_type][obj] = [
                f"An image of {obj} in {style} style" for style in styles[:5]  # 5 styles per object
            ]

    return forgot_prompts, retain_prompts


# Load the Pipeline

def setup_models(cfg):
    dtype = torch.float32

    # Use original diffusers pipelines (no hooks needed for inference)
    if cfg.model_version == 'sdxl':
        from diffusers import StableDiffusionXLPipeline as Pipeline
        print("Loading SDXL pipeline (original diffusers)")
    elif cfg.model_version == 'sd15':
        from diffusers import StableDiffusionPipeline as Pipeline
        print("Loading SD 1.5 pipeline (original diffusers)")
    else:
        raise ValueError(f"Unknown model_version: {cfg.model_version}. Must be 'sdxl' or 'sd15'")

    # Disable NSFW filter for SD 1.5 (needed for nudity concept erasure evaluation)
    extra_kwargs = {}
    if cfg.model_version == 'sd15':
        extra_kwargs['safety_checker'] = None
        extra_kwargs['requires_safety_checker'] = False
        print("Disabled NSFW safety checker for SD 1.5")

    # Load original pipeline
    pipe1 = Pipeline.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map="balanced",
        variant=("fp16" if dtype==torch.float16 else None),
        **extra_kwargs
    )

    # Load fine-tuned pipeline
    pipe2 = Pipeline.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map="balanced",
        variant=("fp16" if dtype==torch.float16 else None),
        **extra_kwargs
    )

    # Load finetune config to get the actual block used during fine-tuning
    # This is important for SD 1.5 where the optimal block may differ from SDXL
    finetune_config_dir = pathlib.Path(cfg.finetune_config_dir)
    model_prefixed_file = finetune_config_dir / f"{cfg.model_version}_{cfg.unlearn_type}.json"
    model_specific_file = finetune_config_dir / cfg.model_version / f"{cfg.unlearn_type}.json"
    model_agnostic_file = finetune_config_dir / f"{cfg.unlearn_type}.json"

    if model_prefixed_file.exists():
        finetune_config_path = model_prefixed_file
        print(f"Using model-prefixed feature config: {finetune_config_path}")
    elif model_specific_file.exists():
        finetune_config_path = model_specific_file
        print(f"Using model-specific feature config (legacy): {finetune_config_path}")
    elif model_agnostic_file.exists():
        finetune_config_path = model_agnostic_file
        print(f"Using model-agnostic feature config: {finetune_config_path}")
    else:
        finetune_config_path = None
        print(f"No feature config found, using code from unlearn config: {cfg.code}")

    # Get the actual block code from feature selection config
    if finetune_config_path:
        with open(finetune_config_path, 'r') as f:
            finetune_config = json.load(f)
            actual_block_code = finetune_config["block"][0]
            print(f"Using block from feature config: {actual_block_code}")
    else:
        actual_block_code = cfg.code
        print(f"Using block from unlearn config: {actual_block_code}")

    # Construct checkpoint path if not provided
    if cfg.checkpoint_path is None or cfg.checkpoint_path == "null":
        # Get lambda_neg_c based on unlearn_type (matching finetune_block.py logic)
        if cfg.unlearn_type == "object_church":
            lambda_neg_c = 1.8
        elif cfg.unlearn_type == "object_parachute":
            lambda_neg_c = 2.0
        else:
            lambda_neg_c = 0.2

        # Try new naming convention with model version first
        checkpoint_filename_new = f"{cfg.model_version}_{cfg.unlearn_type}_99_{lambda_neg_c}_1.0_block.pt"
        checkpoint_path_new = pathlib.Path(cfg.finetune_model_path) / checkpoint_filename_new

        # Fallback to old naming convention without model version
        checkpoint_filename_old = f"{cfg.unlearn_type}_99_{lambda_neg_c}_1.0_block.pt"
        checkpoint_path_old = pathlib.Path(cfg.finetune_model_path) / checkpoint_filename_old

        if checkpoint_path_new.exists():
            checkpoint_path = checkpoint_path_new
        elif checkpoint_path_old.exists():
            checkpoint_path = checkpoint_path_old
            print(f"Warning: Using old checkpoint naming convention. Consider retraining with new version.")
        else:
            raise FileNotFoundError(
                f"Checkpoint not found. Tried:\n"
                f"  - {checkpoint_path_new}\n"
                f"  - {checkpoint_path_old}"
            )
    else:
        checkpoint_path = pathlib.Path(cfg.checkpoint_path)

    # Load fine-tuned checkpoint and inject into the correct block
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")

    # Parse actual_block_code to load into correct block (e.g., "down.2.1", "up.0.1", "mid.0")
    code = actual_block_code
    block_type = code.split(".")[0]

    if block_type == "down":
        block_idx = int(code.split(".")[1])
        attention_idx = int(code.split(".")[2])
        pipe2.unet.down_blocks[block_idx].attentions[attention_idx].load_state_dict(
            checkpoint["model_state_dict"]
        )
        print(f"Loaded fine-tuned weights into down_blocks.{block_idx}.attentions.{attention_idx}")
    elif block_type == "up":
        block_idx = int(code.split(".")[1])
        attention_idx = int(code.split(".")[2])
        pipe2.unet.up_blocks[block_idx].attentions[attention_idx].load_state_dict(
            checkpoint["model_state_dict"]
        )
        print(f"Loaded fine-tuned weights into up_blocks.{block_idx}.attentions.{attention_idx}")
    elif block_type == "mid":
        block_idx = int(code.split(".")[1])
        pipe2.unet.mid_block.attentions[block_idx].load_state_dict(
            checkpoint["model_state_dict"]
        )
        print(f"Loaded fine-tuned weights into mid_block.attentions.{block_idx}")
    else:
        raise ValueError(f"Unsupported block type: {block_type}")

    pipe2.set_progress_bar_config(disable=True)

    return pipe1, pipe2
    

@hydra.main(version_base=None, config_path="conf", config_name="inference")
def main(cfg: DictConfig):

    pipe1, pipe2 = setup_models(cfg)

    print(f"Output directory: {cfg.output_path}")
    print(f"Method: {cfg.method}")
    print(f"Unlearn type: {cfg.unlearn_type}")
    print(f"Number of inference steps: {cfg.num_inference_steps}")
    print(f"Guidance scale: {cfg.guidance_scale}\n")

    # Check if this is an UnlearnCanvas task
    is_unlearncanvas = cfg.unlearn_type.startswith('uc_')

    if is_unlearncanvas and cfg.unlearncanvas_resources_path:
        # Dynamic prompt generation for UnlearnCanvas
        print("UnlearnCanvas task detected - generating test prompts dynamically")

        # Extract target name: try target_name first, then img_class, then derive from unlearn_type
        if hasattr(cfg, 'target_name') and cfg.target_name:
            target_name = cfg.target_name
        elif hasattr(cfg, 'img_class') and cfg.img_class:
            # img_class might be a string or list
            target_name = cfg.img_class if isinstance(cfg.img_class, str) else cfg.img_class[0]
        else:
            # Derive from unlearn_type: uc_object_horses -> Horses
            target_name = cfg.unlearn_type.replace('uc_style_', '').replace('uc_object_', '').replace('_', ' ').title()

        print(f"Target concept: {target_name}")

        forgot_prompts, retain_prompts = generate_unlearncanvas_test_prompts(
            cfg.unlearn_type,
            target_name,
            cfg.unlearncanvas_resources_path
        )
        print(f"Generated {sum(len(v) for v in forgot_prompts[cfg.unlearn_type].values())} forgot prompts")
        print(f"Generated {sum(len(v) for k in retain_prompts[cfg.unlearn_type] for v in retain_prompts[cfg.unlearn_type][k])} retain prompts")
    else:
        # Load from JSON files (original behavior)
        with open(cfg.test_forgot_prompts, 'r') as f:
            forgot_prompts = json.load(f)

        with open(cfg.test_retain_prompts, 'r') as f:
            retain_prompts = json.load(f)

    num=0
    for img_class in forgot_prompts[cfg.unlearn_type]:
        save_path = os.path.join(cfg.output_path, "forgot", cfg.method)
        save_path2 = os.path.join(cfg.output_path, "forgot", "ori")
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path2, exist_ok=True)
        print(f"Processing forgot prompts for class: {img_class}")
        print(f"  Saving fine-tuned images to: {save_path}")
        print(f"  Saving original images to: {save_path2}")
        for i, prompt in enumerate(forgot_prompts[cfg.unlearn_type][img_class]):
            print(f"  Generating image {num+1} with prompt: {prompt[:50]}...")
            images = pipe2(prompt, guidance_scale=cfg.guidance_scale, num_images_per_prompt=1, num_inference_steps=cfg.num_inference_steps, generator=torch.Generator(device="cpu").manual_seed(42)).images
            for k, img in enumerate(images):
                img.save(F"{save_path}/{num}_{k}.png")
            images = pipe1(prompt, guidance_scale=cfg.guidance_scale, num_images_per_prompt=1, num_inference_steps=cfg.num_inference_steps, generator=torch.Generator(device="cpu").manual_seed(42)).images
            for k, img in enumerate(images):
                img.save(F"{save_path2}/{num}_{k}.png")
            num=num+1

    print(f"\n✓ Inference complete! Images saved to {cfg.output_path}")
    print(f"  Total images generated: {num}")

    num=0
    for img_class in retain_prompts[cfg.unlearn_type]:
        # Determine if this is an UnlearnCanvas task (starts with 'uc_')
        is_unlearncanvas = cfg.unlearn_type.startswith('uc_')

        if is_unlearncanvas:
            # UnlearnCanvas: save to unrelated/{class}/
            save_path = os.path.join(cfg.output_path,  "unrelated", cfg.method, img_class)
            save_path2 = os.path.join(cfg.output_path,  "unrelated", "ori", img_class)
        elif cfg.unlearn_type == "nudity" or cfg.unlearn_type == "style_vangogh":
            # Original ConceptACP: save to remove_word/
            save_path = os.path.join(cfg.output_path, "remove_word", cfg.method)
            save_path2 = os.path.join(cfg.output_path,  "remove_word", "ori")
        else:
            # Other object erasure tasks: save to unrelated/{class}/
            save_path = os.path.join(cfg.output_path,  "unrelated", cfg.method, img_class)
            save_path2 = os.path.join(cfg.output_path,  "unrelated", "ori", img_class)

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path2, exist_ok=True)
        print(f"Processing retain_prompts for class: {img_class}")
        print(f"  Saving fine-tuned images to: {save_path}")
        print(f"  Saving original images to: {save_path2}")
        for i, prompt in enumerate(retain_prompts[cfg.unlearn_type][img_class]):
            print(f"  Generating image {num+1} with prompt: {prompt[:50]}...")
            images = pipe2(prompt, guidance_scale=cfg.guidance_scale, num_images_per_prompt=1, num_inference_steps=cfg.num_inference_steps, generator=torch.Generator(device="cpu").manual_seed(42)).images
            for k, img in enumerate(images):
                img.save(F"{save_path}/{num}_{k}.png")
            images = pipe1(prompt, guidance_scale=cfg.guidance_scale, num_images_per_prompt=1, num_inference_steps=cfg.num_inference_steps, generator=torch.Generator(device="cpu").manual_seed(42)).images
            for k, img in enumerate(images):
                img.save(F"{save_path2}/{num}_{k}.png")
            num=num+1

    print(f"\n✓ Inference complete! Images saved to {cfg.output_path}")
    print(f"  Total images generated: {num}")

if __name__ == "__main__":
    main()

