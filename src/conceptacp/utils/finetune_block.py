"""
Efficient block-level fine-tuning.
Trains only the target attention block using pre-computed input/output pairs.
Memory and compute efficient - no need to load the full pipeline!
"""
import torch
from torch.utils.data import DataLoader, IterableDataset
import logging
import json
import glob
import webdataset as wds
import os
import io
import hydra
from omegaconf import DictConfig
import pathlib
import time
import psutil
from diffusers import StableDiffusionXLPipeline

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


class BlockDataset(IterableDataset):
    """Dataset that loads pre-computed block input/output pairs."""

    def __init__(self, tar_path, device="cuda", preload_to_gpu=False):
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Data file not found: {tar_path}")

        self.device = torch.device(device)
        self.preload_to_gpu = preload_to_gpu
        self.cached_samples = None

        # Build the webdataset pipeline
        self.dataset = (
            wds.WebDataset(tar_path, shardshuffle=False, empty_check=False)
            .decode(
                wds.handle_extension(".pth", self._decode_pth),
                wds.handle_extension(".json", self._decode_json),
            )
            .to_tuple("input.pth", "target.pth", "metadata.json")
        )

        # Preload all data if requested
        if preload_to_gpu:
            logger.info(f"Preloading dataset to GPU from {tar_path}...")
            self._preload_to_gpu()

    def _decode_pth(self, data):
        """Decode .pth files with optimizations."""
        buffer = io.BytesIO(data)
        tensor_data = torch.load(buffer, weights_only=False, map_location='cpu')
        return tensor_data

    def _decode_json(self, data):
        """Decode .json files."""
        return json.loads(data.decode())

    def _preload_to_gpu(self):
        """Load entire dataset into GPU memory for fastest training."""
        samples = []
        start_time = time.time()

        # Get GPU memory before loading
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / 1024**3  # GB

        # Load all samples
        for sample in self.dataset:
            inputs, targets, metadata = sample

            # Move inputs to GPU
            inputs_gpu = {
                'hidden_states': inputs['hidden_states'].to(self.device, dtype=torch.float32),
                'encoder_hidden_states': inputs['encoder_hidden_states'].to(self.device, dtype=torch.float32) if inputs['encoder_hidden_states'] is not None else None,
                'attention_mask': inputs['attention_mask'].to(self.device) if inputs['attention_mask'] is not None else None,
            }

            # Move targets to GPU
            targets_gpu = targets.to(self.device, dtype=torch.float32)

            samples.append((inputs_gpu, targets_gpu, metadata))

        self.cached_samples = samples

        # Get GPU memory after loading
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**3  # GB
            mem_used = mem_after - mem_before

        load_time = time.time() - start_time
        logger.info(f"Preloaded {len(samples)} samples to GPU in {load_time:.2f}s")
        logger.info(f"GPU memory used by dataset: {mem_used:.2f}GB")

    def __iter__(self):
        """Iterate through dataset (from GPU cache if preloaded)."""
        if self.cached_samples is not None:
            # Yield from GPU cache
            for sample in self.cached_samples:
                yield sample
        else:
            # Yield from disk (original behavior)
            for sample in self.dataset:
                yield sample


def collate_batch(batch):
    """
    Efficiently collate a batch of samples on CPU with pinned memory.
    GPU transfer happens in the training loop with non-blocking async transfers.
    """
    inputs_list, targets_list, metadata_list = zip(*batch)

    # Check if data is already on GPU (from preloaded dataset)
    if isinstance(inputs_list[0]['hidden_states'], torch.Tensor) and inputs_list[0]['hidden_states'].is_cuda:
        # Data already on GPU - need to squeeze and stack
        hidden_states_list = []
        encoder_hidden_states_list = []
        attention_mask_list = []

        for inp in inputs_list:
            # Squeeze extra dimensions (same logic as CPU path)
            hs = inp['hidden_states']
            while hs.dim() > 3 and hs.shape[0] == 1:
                hs = hs.squeeze(0)
            hidden_states_list.append(hs)

            ehs = inp['encoder_hidden_states']
            if ehs is not None:
                while ehs.dim() > 2 and ehs.shape[0] == 1:
                    ehs = ehs.squeeze(0)
            encoder_hidden_states_list.append(ehs)

            attention_mask_list.append(inp['attention_mask'])

        # Stack on GPU
        hidden_states = torch.stack(hidden_states_list)

        # Handle encoder_hidden_states
        if encoder_hidden_states_list[0] is not None:
            encoder_hidden_states = torch.stack(encoder_hidden_states_list)
        else:
            encoder_hidden_states = None

        # Handle attention_mask
        if attention_mask_list[0] is not None:
            attention_mask = torch.stack(attention_mask_list)
        else:
            attention_mask = None

        # Process targets - squeeze extra dimensions
        processed_targets = []
        for t in targets_list:
            # Remove all singleton dimensions
            while t.dim() > 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.dim() == 5 and t.shape[1] == 1:
                t = t.squeeze(1)
            processed_targets.append(t)

        targets = torch.stack(processed_targets)

        return {
            'hidden_states': hidden_states,
            'encoder_hidden_states': encoder_hidden_states,
            'attention_mask': attention_mask
        }, targets, metadata_list

    # Original CPU path (when not preloaded to GPU)
    # Process targets - need to squeeze extra dimensions
    processed_targets = []
    for t in targets_list:
        t = t.to(dtype=torch.float32)
        # Remove all singleton dimensions (batch=1, timestep=1, etc.)
        while t.dim() > 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() == 5 and t.shape[1] == 1:  # [batch, 1, C, H, W]
            t = t.squeeze(1)
        processed_targets.append(t)

    # Stack targets - should be [C, H, W] per sample → [batch, C, H, W]
    targets = torch.stack(processed_targets)

    # Process inputs - need to handle dict structure
    # Stack all the input components
    hidden_states_list = []
    encoder_hidden_states_list = []
    attention_mask_list = []

    for inp in inputs_list:
        # Each input may have an extra batch dimension (from data generation)
        # Need to squeeze it before stacking
        hs = inp['hidden_states']
        # Remove all leading singleton dimensions
        while hs.dim() > 3 and hs.shape[0] == 1:
            hs = hs.squeeze(0)
        hidden_states_list.append(hs)

        ehs = inp['encoder_hidden_states']
        if ehs is not None:
            while ehs.dim() > 2 and ehs.shape[0] == 1:
                ehs = ehs.squeeze(0)
        encoder_hidden_states_list.append(ehs)

        attention_mask_list.append(inp['attention_mask'])

    # Stack on CPU with contiguous memory layout for faster GPU transfer
    hidden_states = torch.stack(hidden_states_list).to(dtype=torch.float32).contiguous()

    # Handle encoder_hidden_states (might be None)
    if encoder_hidden_states_list[0] is not None:
        encoder_hidden_states = torch.stack(encoder_hidden_states_list).to(dtype=torch.float32).contiguous()
    else:
        encoder_hidden_states = None

    # Handle attention_mask (might be None)
    if attention_mask_list[0] is not None:
        attention_mask = torch.stack(attention_mask_list).contiguous()
    else:
        attention_mask = None

    return {
        'hidden_states': hidden_states,
        'encoder_hidden_states': encoder_hidden_states,
        'attention_mask': attention_mask
    }, targets.contiguous(), metadata_list


def get_attention_block(code, model_path, model_version, device="cuda"):
    """
    Load only the specific attention block from the model.
    Much more memory efficient than loading the full pipeline!

    Args:
        code: Block code (e.g., "down.2.1")
        model_path: Path to the model (local or HuggingFace repo)
        model_version: 'sdxl' or 'sd15'
        device: Device to load the block on
    """
    # We still need to load the full model to extract the block,
    # but we'll immediately delete the rest
    logger.info(f"Loading {model_version.upper()} model from {model_path} to extract block {code}...")

    # Conditional pipeline class based on model version
    if model_version == 'sdxl':
        from diffusers import StableDiffusionXLPipeline as PipelineClass
    elif model_version == 'sd15':
        from diffusers import StableDiffusionPipeline as PipelineClass
    else:
        raise ValueError(f"Unknown model_version: {model_version}. Must be 'sdxl' or 'sd15'")

    # Load with minimal memory footprint
    pipe = PipelineClass.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        variant=None,
        low_cpu_mem_usage=True,
    )

    # Extract the target block
    block_type = code.split(".")[0]
    match block_type:
        case "down":
            block_idx = int(code.split(".")[1])
            attention_idx = int(code.split(".")[2])
            target_block = pipe.unet.down_blocks[block_idx].attentions[attention_idx]
            block_prefix = f"down_blocks.{block_idx}.attentions.{attention_idx}"
        case "mid":
            block_idx = int(code.split(".")[1])
            target_block = pipe.unet.mid_block.attentions[0]
            block_prefix = "mid_block.attentions.0"
        case "up":
            block_idx = int(code.split(".")[1])
            attention_idx = int(code.split(".")[2])
            target_block = pipe.unet.up_blocks[block_idx].attentions[attention_idx]
            block_prefix = f"up_blocks.{block_idx}.attentions.{attention_idx}"
        case _:
            raise ValueError(f"Unsupported block type: {block_type}")

    # Move to device and enable gradients
    target_block = target_block.to(device)
    for param in target_block.parameters():
        param.requires_grad = True

    # Count parameters
    trainable_params = sum(p.numel() for p in target_block.parameters() if p.requires_grad)
    logger.info(f"Loaded block {code} with {trainable_params:,} trainable parameters")

    # Clean up the rest of the model to free memory
    del pipe
    torch.cuda.empty_cache()

    return target_block, block_prefix


def save_checkpoint(cfg, block, block_prefix, epoch, loss):
    """Save only the block state dict."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': block.state_dict(),
        'loss': loss,
        'block_prefix': block_prefix,
    }

    # Get lambda_neg_c based on unlearn_type
    # if cfg.unlearn_type == "object_church":
    #     lambda_neg_c = 1.8
    # elif cfg.unlearn_type == "object_parachute":
    #     lambda_neg_c = 2.0
    # else:
    #     lambda_neg_c = 0.2
    lambda_neg_c=cfg.lambda_neg_c
    logger.info(f"Lambda_neg_c: {lambda_neg_c}")
    target_path = pathlib.Path(cfg.finetune_model_path) / f'{cfg.model_version}_{cfg.unlearn_type}_{epoch}_{lambda_neg_c}_{cfg.make_data_scale}_block.pt'
    target_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, target_path)
    logger.info(f"Saved checkpoint: {target_path}")


def train_block(cfg):
    """Main training loop for block-level fine-tuning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # First, determine which data files to use
    # Load finetune config to get the actual block name used during data generation
    finetune_config_dir = pathlib.Path(cfg.finetune_config_dir)
    model_prefixed_file = finetune_config_dir / f"{cfg.model_version}_{cfg.unlearn_type}.json"
    model_specific_file = finetune_config_dir / cfg.model_version / f"{cfg.unlearn_type}.json"
    model_agnostic_file = finetune_config_dir / f"{cfg.unlearn_type}.json"

    if model_prefixed_file.exists():
        finetune_config_path = model_prefixed_file
        logger.info(f"Using model-prefixed config: {finetune_config_path}")
    elif model_specific_file.exists():
        finetune_config_path = model_specific_file
        logger.info(f"Using model-specific config (legacy): {finetune_config_path}")
    elif model_agnostic_file.exists():
        finetune_config_path = model_agnostic_file
        logger.info(f"Using model-agnostic config: {finetune_config_path}")
    else:
        # Fall back to using cfg.code if no config file exists
        finetune_config_path = None
        logger.info(f"No finetune config found, will use cfg.code")

    if finetune_config_path:
        with open(finetune_config_path, 'r') as f:
            finetune_config = json.load(f)
            actual_block_code = finetune_config["block"][0]
            logger.info(f"Using actual block code from finetune config: {actual_block_code}")
    else:
        actual_block_code = cfg.code
        logger.info(f"Using block code from config: {actual_block_code}")

    # Determine data file paths
    if cfg.get('forgot_file') is not None and cfg.get('retain_file') is not None:
        # Manual specification - use absolute or relative paths
        forgot_path = pathlib.Path(cfg.forgot_file)
        retain_path = pathlib.Path(cfg.retain_file)

        # Make paths absolute if they're relative
        if not forgot_path.is_absolute():
            forgot_path = pathlib.Path(cfg.work_dir) / forgot_path
        if not retain_path.is_absolute():
            retain_path = pathlib.Path(cfg.work_dir) / retain_path

        # Validate files exist
        if not forgot_path.exists():
            raise FileNotFoundError(f"Manually specified forgot file not found: {forgot_path}")
        if not retain_path.exists():
            raise FileNotFoundError(f"Manually specified retain file not found: {retain_path}")

        logger.info("Using manually specified data files:")
        logger.info(f"  Forgot: {forgot_path}")
        logger.info(f"  Retain: {retain_path}")
    else:
        # Automatic file discovery
        forgot_pattern = f"{actual_block_code}_{cfg.unlearn_type}_{cfg.make_data_scale}_block*.tar.gz"
        retain_pattern = f"{actual_block_code}_{cfg.unlearn_type}_retain_block*.tar.gz"

        recons_path = pathlib.Path(cfg.recons_save_path)
        forgot_files = sorted(glob.glob(str(recons_path / forgot_pattern)))
        retain_files = sorted(glob.glob(str(recons_path / retain_pattern)))

        if not forgot_files:
            raise FileNotFoundError(f"No forgot data files found matching pattern: {forgot_pattern} in {recons_path}")
        if not retain_files:
            raise FileNotFoundError(f"No retain data files found matching pattern: {retain_pattern} in {recons_path}")

        forgot_path = pathlib.Path(forgot_files[-1])
        retain_path = pathlib.Path(retain_files[-1])

        logger.info("Using automatically discovered data files:")
        logger.info(f"  Forgot: {forgot_path} (matched {len(forgot_files)} files)")
        logger.info(f"  Retain: {retain_path} (matched {len(retain_files)} files)")

    # Read block metadata from the dataset
    # This is the KEY FIX - read the block info from the data itself!
    logger.info("Reading block metadata from dataset...")
    temp_dataset = (
        wds.WebDataset(forgot_path.as_posix(), shardshuffle=False, empty_check=False)
        .decode(wds.handle_extension(".json", lambda data: json.loads(data.decode())))
    )
    # Get the first sample's metadata
    for sample in temp_dataset:
        if 'metadata.json' in sample:
            metadata = sample['metadata.json']
            block_code_from_data = metadata.get('block_code')
            block_name_from_data = metadata.get('block_name')

            if block_code_from_data:
                logger.info(f"Block metadata from dataset: code={block_code_from_data}, name={block_name_from_data}")
                # Use the block code from the data
                block_code_to_load = block_code_from_data
            else:
                logger.warning("No block metadata found in dataset, falling back to config")
                block_code_to_load = actual_block_code
            break
    else:
        logger.warning("Could not read dataset metadata, falling back to config")
        block_code_to_load = actual_block_code

    # Load the target attention block using the block code from the dataset
    logger.info(f"Loading block: {block_code_to_load}")
    block, block_prefix = get_attention_block(block_code_to_load, cfg.model_path, cfg.model_version, device)

    # # Get lambda_neg_c based on unlearn_type
    # if cfg.unlearn_type == "object_church":
    #     lambda_neg_c = 1.8
    # elif cfg.unlearn_type == "object_parachute":
    #     lambda_neg_c = 2.0
    # else:
    #     lambda_neg_c = 0.2
    lambda_neg_c=cfg.lambda_neg_c
    logger.info(f"Lambda_neg_c: {lambda_neg_c}")

    # Save initial state for comparison
    initial_state = {name: param.clone().detach().cpu() for name, param in block.named_parameters()}

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Log initial memory
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory - CPU: {initial_memory['cpu_mb']:.1f}MB, GPU: {initial_memory.get('gpu_allocated_mb', 0):.1f}MB")

    # Get training config
    batch_size = cfg.get('batch_size', 32)
    num_workers = cfg.get('num_workers', 4)
    prefetch_factor = cfg.get('prefetch_factor', 2) if num_workers > 0 else None
    gradient_accumulation_steps = cfg.get('gradient_accumulation_steps', 1)
    preload_to_gpu = cfg.get('preload_to_gpu', True)  # Default: True for small datasets

    logger.info(f"Training config: batch_size={batch_size}, num_workers={num_workers}, "
                f"prefetch_factor={prefetch_factor}, gradient_accumulation_steps={gradient_accumulation_steps}, "
                f"preload_to_gpu={preload_to_gpu}")

    # Create datasets with optional GPU preloading
    dataset_forgot = BlockDataset(tar_path=forgot_path.as_posix(), device=device, preload_to_gpu=preload_to_gpu)
    dataset_retain = BlockDataset(tar_path=retain_path.as_posix(), device=device, preload_to_gpu=preload_to_gpu)

    # When preloading to GPU, we don't need workers/pinning/prefetch
    if preload_to_gpu:
        num_workers = 0
        pin_memory = False
        prefetch_factor = None
        logger.info("GPU preloading enabled - disabling CPU workers for optimal performance")
    else:
        pin_memory = True

    dataloader_forgot = DataLoader(
        dataset_forgot,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    dataloader_retain = DataLoader(
        dataset_retain,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-5, weight_decay=0.01)

    num_epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Setup mixed precision training
    use_amp = cfg.get('use_amp', True)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    logger.info(f"Using mixed precision (AMP): {use_amp}")

    logger.info(f"Starting training for {num_epochs} epochs")

    # Track overall training time
    training_start_time = time.time()
    epoch_times = []

    # Training loop
    for epoch in range(num_epochs):
        block.train()
        total_loss = 0
        total_loss_forgot = 0
        total_loss_retain = 0
        num_batches = 0

        epoch_start = time.time()

        # Zip both dataloaders together
        for batch_idx, ((inputs_forgot, targets_forgot, _), (inputs_retain, targets_retain, _)) in enumerate(
            zip(dataloader_forgot, dataloader_retain)
        ):
            # Move data to GPU only if not already there (when preload_to_gpu=False)
            if not targets_forgot.is_cuda:
                targets_forgot = targets_forgot.to(device, non_blocking=True)
                targets_retain = targets_retain.to(device, non_blocking=True)

                inputs_forgot_gpu = {
                    'hidden_states': inputs_forgot['hidden_states'].to(device, non_blocking=True),
                    'encoder_hidden_states': inputs_forgot['encoder_hidden_states'].to(device, non_blocking=True) if inputs_forgot['encoder_hidden_states'] is not None else None,
                    'attention_mask': inputs_forgot['attention_mask'].to(device, non_blocking=True) if inputs_forgot['attention_mask'] is not None else None,
                }

                inputs_retain_gpu = {
                    'hidden_states': inputs_retain['hidden_states'].to(device, non_blocking=True),
                    'encoder_hidden_states': inputs_retain['encoder_hidden_states'].to(device, non_blocking=True) if inputs_retain['encoder_hidden_states'] is not None else None,
                    'attention_mask': inputs_retain['attention_mask'].to(device, non_blocking=True) if inputs_retain['attention_mask'] is not None else None,
                }
            else:
                # Data already on GPU from preloading - use directly
                inputs_forgot_gpu = inputs_forgot
                inputs_retain_gpu = inputs_retain

            # Use automatic mixed precision if enabled
            with torch.amp.autocast('cuda', enabled=use_amp):
                # Forward pass for forgot samples (concept to erase)
                # print(f"{inputs_forgot_gpu['hidden_states'].shape=}")
                # print(f"{inputs_forgot_gpu['encoder_hidden_states'].shape=}")
                # print(f"{inputs_forgot_gpu['attention_mask']=}")  # May be None
                output_forgot = block(
                    inputs_forgot_gpu['hidden_states'],
                    encoder_hidden_states=inputs_forgot_gpu['encoder_hidden_states'],
                    attention_mask=inputs_forgot_gpu['attention_mask'],
                )

                # Extract tensor from output (handle different return types)
                if hasattr(output_forgot, 'sample'):
                    output_forgot = output_forgot.sample
                elif isinstance(output_forgot, tuple):
                    output_forgot = output_forgot[0]

                if targets_forgot.dim() == 5:
                    targets_forgot = targets_forgot.squeeze(1)
                if targets_forgot.dim() == 4 and targets_forgot.shape[1] == 1:
                    # Squeeze timestep dimension if it's 1
                    targets_forgot = targets_forgot.squeeze(1)

                # Handle shape mismatch between spatial (SDXL) and sequence (SD 1.5) formats
                if output_forgot.dim() == 4 and targets_forgot.dim() == 3:
                    # Output is spatial [batch, features, height, width]
                    # Target is sequence [batch, seq_len, features]
                    # Convert output to sequence format
                    output_forgot = output_forgot.flatten(2, 3).permute(0, 2, 1)  # [batch, seq_len, features]
                elif output_forgot.dim() == 3 and targets_forgot.dim() == 4:
                    # Output is sequence [batch, seq_len, features]
                    # Target is spatial [batch, features, height, width]
                    # Convert output to spatial format
                    batch_size, seq_len, features = output_forgot.shape
                    height = width = int(seq_len ** 0.5)
                    output_forgot = output_forgot.permute(0, 2, 1).reshape(batch_size, features, height, width)

                # Compute loss for forgot batch
                loss_forgot = torch.nn.functional.mse_loss(output_forgot, targets_forgot)

                # Forward pass for retain samples (preserve these concepts)
                output_retain = block(
                    inputs_retain_gpu['hidden_states'],
                    encoder_hidden_states=inputs_retain_gpu['encoder_hidden_states'],
                    attention_mask=inputs_retain_gpu['attention_mask'],
                )

                # Extract tensor from output
                if hasattr(output_retain, 'sample'):
                    output_retain = output_retain.sample
                elif isinstance(output_retain, tuple):
                    output_retain = output_retain[0]

                if targets_retain.dim() == 5:
                    targets_retain = targets_retain.squeeze(1)
                if targets_retain.dim() == 4 and targets_retain.shape[1] == 1:
                    # Squeeze timestep dimension if it's 1
                    targets_retain = targets_retain.squeeze(1)

                # Handle shape mismatch between spatial (SDXL) and sequence (SD 1.5) formats
                if output_retain.dim() == 4 and targets_retain.dim() == 3:
                    # Output is spatial [batch, features, height, width]
                    # Target is sequence [batch, seq_len, features]
                    # Convert output to sequence format
                    output_retain = output_retain.flatten(2, 3).permute(0, 2, 1)  # [batch, seq_len, features]
                elif output_retain.dim() == 3 and targets_retain.dim() == 4:
                    # Output is sequence [batch, seq_len, features]
                    # Target is spatial [batch, features, height, width]
                    # Convert output to spatial format
                    batch_size, seq_len, features = output_retain.shape
                    height = width = int(seq_len ** 0.5)
                    output_retain = output_retain.permute(0, 2, 1).reshape(batch_size, features, height, width)

                # Compute loss for retain batch
                loss_retain = torch.nn.functional.mse_loss(output_retain, targets_retain)

                # Combined loss
                loss = (loss_forgot + lambda_neg_c * loss_retain) / gradient_accumulation_steps

            # Backward pass with scaling if using AMP
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Only step optimizer every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(block.parameters(), max_norm=1.0)
                    # Step optimizer with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(block.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            total_loss_forgot += loss_forgot.item()
            total_loss_retain += loss_retain.item()
            num_batches += 1

        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_loss_forgot = total_loss_forgot / num_batches if num_batches > 0 else 0
        avg_loss_retain = total_loss_retain / num_batches if num_batches > 0 else 0

        # Get current memory usage
        current_memory = get_memory_usage()

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} (forgot: {avg_loss_forgot:.4f}, retain: {avg_loss_retain:.4f}) | "
            f"Time: {epoch_time:.2f}s | "
            f"GPU: {current_memory.get('gpu_allocated_mb', 0):.0f}MB"
        )

        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            save_checkpoint(cfg, block, block_prefix, epoch, avg_loss)

    # Training complete - compute final statistics
    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    # Get final memory usage
    final_memory = get_memory_usage()
    peak_memory = final_memory.get('gpu_max_mb', 0)

    # Compute parameter changes
    final_state = {name: param.clone().detach().cpu() for name, param in block.named_parameters()}
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

    return block


@hydra.main(version_base=None, config_path="conf", config_name="finetune_block")
def main(cfg: DictConfig):
    train_block(cfg)


if __name__ == "__main__":
    main()
