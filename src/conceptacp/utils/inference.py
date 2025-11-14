import json
import pathlib
from conceptacp.sae.SDLens import HookedStableDiffusionXLPipeline
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig
# from datasets import load_dataset

device = torch.device("cuda:1")

# Load the Pipeline

def setup_models(cfg):
    dtype = torch.float32
    pipe1 = HookedStableDiffusionXLPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map="balanced",
        variant=("fp16" if dtype==torch.float16 else None)
    )

    pipe2= HookedStableDiffusionXLPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map="balanced",
        variant=("fp16" if dtype==torch.float16 else None)
    )

    checkpoint = torch.load(cfg.finetune_model_path, map_location="cuda")
    if cfg.data.unlearn_type=="style_vangogh":
        pipe2.pipe.unet.up_blocks[0].attentions[1].load_state_dict(checkpoint["model_state_dict"])
    else:
        pipe2.pipe.unet.down_blocks[2].attentions[1].load_state_dict(checkpoint["model_state_dict"])
    pipe2.pipe.set_progress_bar_config(disable=True)

    return pipe1, pipe2
    

@hydra.main(version_base=None, config_path="conf", config_name="inference")
def main(cfg: DictConfig):

    pipe1, pipe2 = setup_models(cfg)
    

    with open(cfg.data.test_forgot_prompts, 'r') as f:
        forgot_prompts = json.load(f)

    with open(cfg.data.test_retain_prompts, 'r') as f:
        retain_prompts = json.load(f)
    
    num=0
    for img_class in forgot_prompts[cfg.data.unlearn_type]:
        save_path=f"/output_images/{cfg.data.unlearn_type}/forgot/{cfg.method}"
        save_path2=f"/output_images/{cfg.data.unlearn_type}/forgot/ori"
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path2, exist_ok=True)
        for i, prompt in enumerate(forgot_prompts[cfg.data.unlearn_type][img_class]):    
            images = pipe2(prompt, guidance_scale=0.0,num_images_per_prompt=1, num_inference_steps=cfg.num_inference_steps, generator=torch.Generator(device="cpu").manual_seed(42)).images
            for k, img in enumerate(images):
                img.save(F"{save_path}/{num}_{k}.png")
            images = pipe1(prompt, guidance_scale=0.0,num_images_per_prompt=1, num_inference_steps=cfg.num_inference_steps, generator=torch.Generator(device="cpu").manual_seed(42)).images
            for k, img in enumerate(images):
                img.save(F"{save_path2}/{num}_{k}.png")
            num=num+1

if __name__ == "__main__":
    main()

