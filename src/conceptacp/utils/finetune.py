#%%
import torch
from diffusers import StableDiffusionXLPipeline
from conceptacp.sae.SDLens import HookedStableDiffusionXLPipeline
from torch.utils.data import DataLoader
import logging
import json
import webdataset as wds
import os
from torch.utils.data import IterableDataset
import io
import hydra
from omegaconf import DictConfig
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def unfreeze_layer(pipe, code):
    block_type = code.split(".")[0]
    match block_type:
        case "down":
            block_idx = int(code.split(".")[1])
            attention_idx = int(code.split(".")[2])
            target_layer = pipe.unet.down_blocks[block_idx].attentions[attention_idx]
        case "mid":
            block_idx = int(code.split(".")[1])
            target_layer = pipe.unet.mid_block.attentions[0]
        case "up":
            block_idx = int(code.split(".")[1])
            attention_idx = int(code.split(".")[2])
            target_layer = pipe.unet.up_blocks[block_idx].attentions[attention_idx]
        case _:
            raise ValueError(f"Unsupported block type: {block_type}")
    for param in target_layer.parameters():
        param.requires_grad = True
    
def setup_model(cfg):
    model_id = cfg.model_id
    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     model_id, 
    #     torch_dtype=torch.float16,
    #     variant="fp16"
    # )
    dtype = torch.float32
    hooked_pipe = HookedStableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="balanced",
        variant=("fp16" if dtype==torch.float16 else None)
    )
    pipe = hooked_pipe.pipe
    pipe.set_progress_bar_config(disable=True)
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    for component in [pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]:
        for param in component.parameters():
            param.requires_grad = False
    
    unfreeze_layer(pipe, cfg.data.code)
    
    trainable_params = 0
    for name, param in pipe.unet.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            logger.info(f"params: {name}")
    logger.info(f"trainable_params: {trainable_params:,}")

    # for name, module in pipe.unet.named_modules():
    #     if "down_blocks.2" in name:
    #         print(f"Layer: {name}\nType: {type(module)}\n")
    
    return hooked_pipe


def calculate_loss(outputs, targets):
    return torch.nn.functional.mse_loss(outputs, targets)

def save_checkpoint(cfg, pipe, epoch, loss):
    if cfg.data.unlearn_type=="style_vangogh":
        block_prefix = "up_blocks.0.attentions.1"
    else:
        block_prefix = "down_blocks.2.attentions.1"
    full_state_dict = pipe.pipe.unet.state_dict()

    block_state_dict = {
        k.replace(f"{block_prefix}.", ""): v 
        for k, v in full_state_dict.items() 
        if k.startswith(block_prefix)
    }
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': block_state_dict,
        'loss': loss,
        'block_prefix': block_prefix,  
    }
    if cfg.data.unlearn_type=="object_church":
        lambda_neg_c=1.8
    elif cfg.data.unlearn_type=="object_parachute":
        lambda_neg_c=2.0
    else:
        lambda_neg_c=0.2
    target_path = pathlib.Path(cfg.finetune_model_path) / f'{cfg.data.unlearn_type}_{epoch}_{lambda_neg_c}_{cfg.data.make_data_scale}.pt'
    target_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, target_path)
    logger.info(f"save checkpoint: {target_path}, {len(block_state_dict)}parameters")

class FineTuneDataset(IterableDataset):
    def __init__(self, tar_paths, batch_size=32, device="cuda"):

        self._validate_paths(tar_paths)
        
        self.dataset = (
            wds.WebDataset(tar_paths)
            .decode(
                wds.handle_extension(".pth", self._decode_pth),
                wds.handle_extension(".json", self._decode_json),
            )
            .to_tuple("gen_args.json", "recons.pth")
            .batched(batch_size, partial=False)
        )
        
        self.device = torch.device(device)
        
    def _validate_paths(self, paths):
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"file missing: {missing}")

    def _decode_pth(self, data):
        return torch.load(io.BytesIO(data)).to(self.device)
    
    def _decode_json(self, data):
        return json.loads(data.decode())
    
    def __iter__(self):
        for batch in self.dataset:
            metadata_batch, recons_batch = batch
            prompts = [meta['prompt'] for meta in metadata_batch]
            yield prompts, recons_batch

class ShapeCapture:
        def __init__(self):
            self.shape = None  

        def debug_hook(self, module, input, output):
            if isinstance(output, tuple):
                self.shape = output[0].shape  
                self.tensor = output[0]
            else:
                self.shape = output.shape

def finetune(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FineTuneDataset(tar_paths=[(pathlib.Path(cfg.data.recons_save_path) / f"{cfg.data.code}_{cfg.data.unlearn_type}_{cfg.data.make_data_scale}.tar.gz").as_posix()], batch_size=4)
    dataset_retain = FineTuneDataset(tar_paths=[(pathlib.Path(cfg.data.recons_save_path) / f"{cfg.data.code}_{cfg.data.unlearn_type}_retain.tar.gz").as_posix()], batch_size=4)
    pipe = setup_model(cfg)
    logger.info(f"use device: {device}")
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=None, 
        # num_workers=4,    
        pin_memory=True,   
        # persistent_workers=True,  
        shuffle=False   
    )

    train_dataloader_retain = DataLoader(
        dataset=dataset_retain,
        batch_size=None, 
        # num_workers=4,   
        pin_memory=True,   
        # persistent_workers=True,  
        shuffle=False   
    )
    
    num_epochs=100
    if cfg.data.unlearn_type=="object_church":
        lambda_neg_c=1.8
    elif cfg.data.unlearn_type=="object_parachute":
        lambda_neg_c=2.0
    else:
        lambda_neg_c=0.2
    # lambda_neg_c = cfg.lambda_neg_c
    shape_capturer = ShapeCapture()

    optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, pipe.pipe.unet.parameters()),
            lr=1e-5,
            weight_decay=0.01
        )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs
    )
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_loss_c = 0  
        total_loss_neg_c = 0  
        # for prompts, target_features in train_dataloader:
        for (prompts, target_features), (prompts_neg_c, target_features_neg_c) in zip(train_dataloader, train_dataloader_retain):
            target_features=target_features.permute(0,3,1,2).contiguous().to(device)
            target_features = target_features.to(torch.float32)
            optimizer.zero_grad()

            with torch.enable_grad():
                output= pipe.run_with_hooks_grad(
                    prompt=prompts,
                    position_hook_dict={
                        cfg.data.code_to_block[cfg.data.code]:shape_capturer.debug_hook

                    },
                    num_inference_steps=1,
                    output_type="latent",
                    guidance_scale=0.0, 
                    generator=torch.Generator(device="cuda").manual_seed(42)
                )

            captured_shape = shape_capturer.shape
            captured_tensor = shape_capturer.tensor
            loss_c = torch.nn.functional.mse_loss(captured_tensor,target_features)
            # loss.backward()

            target_features_neg_c = target_features_neg_c.permute(0, 3, 1, 2).contiguous().to(device).to(torch.float32) 
            with torch.enable_grad():
                output_neg_c = pipe.run_with_hooks_grad(
                    prompt=prompts_neg_c, 
                    position_hook_dict={
                        cfg.data.code_to_block[cfg.data.code]: shape_capturer.debug_hook
                    },
                    num_inference_steps=1,
                    output_type="latent",
                    guidance_scale=0.0,
                    generator=torch.Generator(device="cuda").manual_seed(42)
                )
            captured_tensor_neg_c = shape_capturer.tensor
            loss_neg_c = torch.nn.functional.mse_loss(captured_tensor_neg_c, target_features_neg_c)
            loss = loss_c + lambda_neg_c * loss_neg_c
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, pipe.pipe.unet.parameters()),
                max_norm=1.0
            )


            optimizer.step()

            total_loss += loss.item()

        scheduler.step()


        avg_loss = total_loss / 100
        avg_loss_c = total_loss_c / 50
        avg_loss_neg_c = total_loss_neg_c / 50


        logger.info(f"Epoch {epoch+1}/{num_epochs}, avg loss: {avg_loss:.4f}")


        if (epoch + 1) % 100 == 0:
            save_checkpoint(cfg, pipe, epoch, avg_loss)

    return pipe
# %%


@hydra.main(version_base=None, config_path="conf", config_name="finetune")
def main(cfg: DictConfig):
    finetune(cfg)

if __name__ == "__main__":
    main()
# %%