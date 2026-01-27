#Prompt->Latent finetuning
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
import time
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024  # MB

    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        gpu_mem_max = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        return {
            'cpu_mb': cpu_mem,
            'gpu_allocated_mb': gpu_mem_allocated,
            'gpu_reserved_mb': gpu_mem_reserved,
            'gpu_max_mb': gpu_mem_max
        }
    else:
        return {'cpu_mb': cpu_mem}


def compute_parameter_changes(initial_state, final_state):
    """Compute statistics about parameter changes."""
    total_params = 0
    changed_params = 0
    total_change = 0.0
    max_change = 0.0
    max_change_param = None

    for name in initial_state.keys():
        initial = initial_state[name]
        final = final_state[name]

        param_count = initial.numel()
        total_params += param_count

        # Compute parameter-wise changes
        diff = torch.abs(final - initial)
        param_change = diff.sum().item()
        total_change += param_change

        # Count significantly changed parameters (> 1e-6)
        changed_mask = diff > 1e-6
        changed_params += changed_mask.sum().item()

        # Track maximum change
        max_diff = diff.max().item()
        if max_diff > max_change:
            max_change = max_diff
            max_change_param = name

    avg_change = total_change / total_params if total_params > 0 else 0
    pct_changed = (changed_params / total_params * 100) if total_params > 0 else 0

    return {
        'total_params': total_params,
        'changed_params': changed_params,
        'pct_changed': pct_changed,
        'avg_change': avg_change,
        'max_change': max_change,
        'max_change_param': max_change_param
    }


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
    model_path = cfg.data.model_path
    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.float16,
    #     variant="fp16"
    # )
    dtype = torch.float32
    hooked_pipe = HookedStableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        # device_map="balanced",
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

    # Save initial state for comparison
    initial_state = {name: param.clone().detach().cpu()
                    for name, param in pipe.pipe.unet.named_parameters()
                    if param.requires_grad}

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Log initial memory
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory - CPU: {initial_memory['cpu_mb']:.1f}MB, GPU: {initial_memory.get('gpu_allocated_mb', 0):.1f}MB")

    # Track training time
    training_start_time = time.time()
    epoch_times = []

    for epoch in range(num_epochs):
        total_loss = 0
        total_loss_c = 0
        total_loss_neg_c = 0
        epoch_start = time.time()
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

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = total_loss / 100
        avg_loss_c = total_loss_c / 50
        avg_loss_neg_c = total_loss_neg_c / 50

        # Get current memory usage
        current_memory = get_memory_usage()

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.2f}s | "
            f"GPU: {current_memory.get('gpu_allocated_mb', 0):.0f}MB"
        )


        if (epoch + 1) % 100 == 0:
            save_checkpoint(cfg, pipe, epoch, avg_loss)

    # Training complete - compute final statistics
    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    # Get final memory usage
    final_memory = get_memory_usage()
    peak_memory = final_memory.get('gpu_max_mb', 0)

    # Compute parameter changes
    final_state = {name: param.clone().detach().cpu()
                   for name, param in pipe.pipe.unet.named_parameters()
                   if param.requires_grad}
    param_stats = compute_parameter_changes(initial_state, final_state)

    # Log final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total training time: {total_training_time/60:.2f} minutes ({total_training_time:.1f}s)")
    logger.info(f"Average time per epoch: {avg_epoch_time:.2f}s")
    logger.info(f"Total epochs: {num_epochs}")
    logger.info("")
    logger.info("Memory Usage:")
    logger.info(f"  Peak GPU memory: {peak_memory:.1f}MB")
    logger.info(f"  Final GPU memory: {final_memory.get('gpu_allocated_mb', 0):.1f}MB")
    logger.info(f"  CPU memory: {final_memory['cpu_mb']:.1f}MB")
    logger.info("")
    logger.info("Parameter Changes:")
    logger.info(f"  Total parameters: {param_stats['total_params']:,}")
    logger.info(f"  Changed parameters: {param_stats['changed_params']:,} ({param_stats['pct_changed']:.2f}%)")
    logger.info(f"  Average parameter change: {param_stats['avg_change']:.2e}")
    logger.info(f"  Maximum parameter change: {param_stats['max_change']:.2e}")
    logger.info(f"  Layer with max change: {param_stats['max_change_param']}")
    logger.info("="*80 + "\n")

    return pipe
# %%


@hydra.main(version_base=None, config_path="conf", config_name="finetune")
def main(cfg: DictConfig):
    finetune(cfg)

if __name__ == "__main__":
    main()
# %%