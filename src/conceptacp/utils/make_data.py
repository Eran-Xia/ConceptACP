from conceptacp.sae.SDLens import HookedStableDiffusionXLPipeline
from conceptacp.sae.SAE import SparseAutoencoder
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig
import pathlib
import json
import webdataset as wds

def scale_features(sparse_maps, feature_scales):
    for idx, scale in feature_scales.items():
        sparse_maps[:,:,idx] = sparse_maps[:,:,idx] * scale
    return sparse_maps

@hydra.main(config_path="conf/data", config_name="default")
def main(cfg: DictConfig):
    device = torch.device("cuda:0")

    dtype = torch.float32

    pipe = HookedStableDiffusionXLPipeline.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map="balanced",
        variant=("fp16" if dtype==torch.float16 else None)
    )

    pipe.set_progress_bar_config(disable=True)
    # Load the SAEs
 
    path_to_checkpoints = cfg.sae_checkpoints

    saes_dict = {}
    means_dict = {}

    for code, block in cfg.code_to_block.items():
        sae = SparseAutoencoder.load_from_disk(
            os.path.join(path_to_checkpoints, f"{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final"),
        )
        means = torch.load(
            os.path.join(path_to_checkpoints, f"{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final", "mean.pt"),
            weights_only=True
        )
        saes_dict[code] = sae.to('cuda', dtype=dtype)
        means_dict[code] = means.to('cuda', dtype=dtype)

    with open(cfg.retain_prompts, 'r') as f:
        retain = json.load(f)
    with open(cfg.forgot_prompts, 'r') as f:
        forgot = json.load(f)
    
    forgot_prompts=[]
    for img_class in forgot[cfg.unlearn_type]:
        for prompt in forgot[cfg.unlearn_type][img_class]:
            forgot_prompts.append(prompt)

    retain_prompts=[]
    for img_class in retain[cfg.unlearn_type]:
        for prompt in retain[cfg.unlearn_type][img_class]:
            retain_prompts.append(prompt)

    def generate_sparse_maps(cache, code): 
        block = cfg.code_to_block[code] 
        sae = saes_dict[code] 
        diff = cache["output"][block].to(device) - cache["input"][block].to(device) 
        diff = diff.permute(0, 1, 3, 4, 2).squeeze(0).squeeze(0) 
        with torch.no_grad():
            sparse_maps = sae.encode(diff) 
        return sparse_maps    
    
    finetune_config_dir = cfg.finetune_config_dir
    finetune_config_file = pathlib.Path(finetune_config_dir) / (cfg.unlearn_type + ".json")
    with open(finetune_config_file, 'r') as f:
        finetune_config = json.load(f)
        feature_scales = {}
        for feature, scale in zip(finetune_config["finetune_feature"], finetune_config["finetune_strength"]):
            feature_scales[feature] = scale*cfg.make_data_scale
        finetune_block = finetune_config["block"][0]
        finetune_type = finetune_config["unlearn_type"]
    
    recons_save_path=cfg.recons_save_path
    os.makedirs(recons_save_path, exist_ok=True)
    recons_writers = {
        block: wds.TarWriter(f'{recons_save_path}/{finetune_block}_{finetune_type}_{cfg.make_data_scale}.tar.gz')
    }
    sae = saes_dict[finetune_block]
    num=0
    for forgot_prompt in forgot_prompts:
        num=num+1
        output, cache = pipe.run_with_cache(
            forgot_prompt,
            positions_to_cache=list(cfg.code_to_block.values()), 
            save_input=True, 
            save_output=True,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=0.0, 
            generator=torch.Generator(device="cpu").manual_seed(42) 
        )
        # sparse_maps = generate_sparse_maps(cache, cfg.code)
        sparse_maps = generate_sparse_maps(cache, finetune_block)
        sparse_maps = scale_features(sparse_maps, feature_scales)
        recons=sae.decoder(sparse_maps)+sae.pre_bias
        recons=recons.to(device)
        # print(recons.shape)
        # exit()

        kwargs_to_save = {
                'prompt': forgot_prompt,
                'positions_to_cache': dict(cfg.code_to_block),
                'save_input': True,
                'save_output': True,
                'num_inference_steps': cfg.num_inference_steps,
                'guidance_scale': 0.0,
                'output_type': 'pil'
            }
        

        for block, writer in recons_writers.items():
            writer.write({
                "__key__": f"sample_{num}",
                "recons.pth": recons.half().cpu(),
                "gen_args.json": kwargs_to_save
            })


    for block, writer in recons_writers.items():
        writer.close()


    recons_writers2 = {
        block: wds.TarWriter(f'{recons_save_path}/{finetune_block}_{finetune_type}_retain.tar.gz')
    }
    
    num=0
    for retain_prompt in retain_prompts:
        num=num+1
        output, cache = pipe.run_with_cache(
            retain_prompt,
            positions_to_cache=list(cfg.code_to_block.values()),
            save_input=True, 
            save_output=True,
            num_inference_steps=cfg.num_inference_steps, 
            guidance_scale=0.0,
            generator=torch.Generator(device="cpu").manual_seed(42) 
        )
        block = cfg.code_to_block[finetune_block]
        recons = cache["output"][block].to(device) 
        recons = recons.permute(0, 1, 3, 4, 2).squeeze(0).squeeze(0)

        kwargs_to_save = {
                'prompt': retain_prompt,
                'positions_to_cache': dict(cfg.code_to_block),
                'save_input': True,
                'save_output': True,
                'num_inference_steps': cfg.num_inference_steps,
                'guidance_scale': 0.0,
                'output_type': 'pil'
            }

        for block, writer in recons_writers2.items():
            writer.write({
                "__key__": f"sample_{num}",
                "recons.pth": recons.half().cpu(),
                "gen_args.json": kwargs_to_save
            })


    for block, writer in recons_writers2.items():
        writer.close()
        
if __name__ == "__main__":
    main()