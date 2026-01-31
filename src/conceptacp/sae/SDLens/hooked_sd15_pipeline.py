"""
Hooked Stable Diffusion 1.5 Pipeline for ConceptACP
Adapted from SAeUron's hooked_sd_noised_pipeline.py
"""

from typing import Callable, Dict, List, Optional, Union

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)

from .hooked_scheduler_sd15 import HookedNoiseScheduler


def retrieve(io, unconditional: bool = False):
    """Extract conditional or unconditional output from hooked tensor.

    When guidance_scale > 1.0, the batch contains both unconditional and conditional outputs
    concatenated along the batch dimension, so we chunk it.
    When guidance_scale <= 1.0, we only have conditional output, so return as-is.
    """
    if isinstance(io, tuple):
        if len(io) == 1:
            io = io[0].detach().cpu()
        else:
            raise ValueError("A tuple should have length of 1")
    elif isinstance(io, torch.Tensor):
        io = io.detach().cpu()
    else:
        raise ValueError("Input/Output must be a tensor, or 1-element tuple")

    # Check if batch dimension can be split (i.e., classifier-free guidance is enabled)
    # When guidance_scale > 1.0, batch_size is 2 * num_prompts
    # When guidance_scale <= 1.0, batch_size is just num_prompts
    batch_size = io.shape[0]
    if batch_size % 2 == 0 and batch_size >= 2:
        # Batch is doubled - split into unconditional and conditional
        io_uncond, io_cond = io.chunk(2)
        if unconditional:
            return io_uncond
        return io_cond
    else:
        # Batch is not doubled (guidance_scale <= 1.0) - return as-is
        # This is the conditional output
        return io


class HookedStableDiffusionPipeline:
    """
    Hooked wrapper for Stable Diffusion 1.5 pipeline.
    Provides activation caching and hook registration capabilities.
    """

    parent_cls = StableDiffusionPipeline
    pipe = None

    def __init__(self, pipe: parent_cls, use_hooked_scheduler: bool = False):
        if use_hooked_scheduler:
            pipe.scheduler = HookedNoiseScheduler(pipe.scheduler)
        # Preserve the default scheduler from the loaded pipeline
        scheduler_name = pipe.scheduler.__class__.__name__
        print(f"Using default scheduler: {scheduler_name}")
        self.__dict__["pipe"] = pipe
        self.use_hooked_scheduler = use_hooked_scheduler

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(cls.parent_cls.from_pretrained(*args, **kwargs))

    @torch.no_grad()
    def run_with_hooks(
        self,
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        """
        Run the pipeline with hooks at specified positions.
        Returns the final output.

        Args:
            position_hook_dict: A dictionary mapping positions to hooks.
            prompt: Text prompt(s) to condition generation
            num_images_per_prompt: Number of images per prompt
            device: Device to run on
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            generator: Random generator for reproducibility
            latents: Optional initial latents
            output_type: Output format ('pil' or 'latent')
        """
        hooks = []
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]

        try:
            prompt_embeds, timesteps, latents, extra_step_kwargs, added_cond_kwargs = (
                self._prepare_prompt(
                    prompt,
                    device,
                    num_images_per_prompt,
                    guidance_scale,
                    num_inference_steps,
                    generator,
                    latents,
                )
            )

            latents = self._denoise_loop(
                timesteps,
                latents,
                guidance_scale,
                extra_step_kwargs,
                added_cond_kwargs,
                prompt_embeds,
            )
            image = self._postprocess_latents(latents, output_type, generator)
        finally:
            for hook in hooks:
                hook.remove()
            if self.use_hooked_scheduler:
                self.pipe.scheduler.pre_hooks = []
                self.pipe.scheduler.post_hooks = []

        return image

    @torch.no_grad()
    def run_with_cache(
        self,
        prompt: Union[str, List[str]] = None,
        positions_to_cache: List[str] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        save_input: bool = True,  # Changed default to True for ConceptACP compatibility
        save_output: bool = True,
        unconditional: bool = False,
        **kwargs,
    ):
        """
        Run the pipeline with caching at specified positions.
        Captures activations across all timesteps.

        Args:
            positions_to_cache: List of positions to cache (e.g., ['unet.up_blocks.1.attentions.1'])
            save_input: If True, caches inputs at specified positions
            save_output: If True, caches outputs at specified positions
            unconditional: If True, returns unconditional activations

        Returns:
            image: Generated images
            cache_dict: Dictionary with 'input' and/or 'output' keys containing cached activations
                        Shape: (batch, timesteps, spatial_patches, d_in)
        """
        cache_input, cache_output = (
            dict() if save_input else None,
            dict() if save_output else None,
        )
        hooks = [
            self._register_cache_hook(
                position, cache_input, cache_output, unconditional
            )
            for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]

        prompt_embeds, timesteps, latents, extra_step_kwargs, added_cond_kwargs = (
            self._prepare_prompt(
                prompt,
                device,
                num_images_per_prompt,
                guidance_scale,
                num_inference_steps,
                generator,
                latents,
            )
        )

        latents = self._denoise_loop(
            timesteps,
            latents,
            guidance_scale,
            extra_step_kwargs,
            added_cond_kwargs,
            prompt_embeds,
        )

        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []

        # Stack cached activations across timesteps
        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict["input"] = cache_input

        if save_output:
            for position, block in cache_output.items():
                cache_output[position] = torch.stack(block, dim=1)
            cache_dict["output"] = cache_output

        image = self._postprocess_latents(latents, output_type, generator)

        return image, cache_dict

    def run_with_hooks_grad(
        self,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "latent",
        **kwargs,
    ):
        """
        Run pipeline with hooks and gradient tracking enabled.
        Used for fine-tuning specific layers.
        """
        hooks = []
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]

        try:
            prompt_embeds, timesteps, latents, extra_step_kwargs, added_cond_kwargs = (
                self._prepare_prompt(
                    prompt,
                    device,
                    num_images_per_prompt,
                    guidance_scale,
                    num_inference_steps,
                    generator,
                    latents,
                )
            )

            latents = self._denoise_loop(
                timesteps,
                latents,
                guidance_scale,
                extra_step_kwargs,
                added_cond_kwargs,
                prompt_embeds,
            )

            image = self._postprocess_latents(latents, output_type, generator)
        finally:
            for hook in hooks:
                hook.remove()
            if self.use_hooked_scheduler:
                self.pipe.scheduler.pre_hooks = []
                self.pipe.scheduler.post_hooks = []

        return image

    def _locate_block(self, position: str):
        """
        Navigate to block using dot notation.
        Example: 'unet.up_blocks.1.attentions.1'
        """
        block = self.pipe
        for step in position.split("."):
            if step.isdigit():
                step = int(step)
                block = block[step]
            else:
                block = getattr(block, step)
        return block

    def _register_cache_hook(
        self,
        position: str,
        cache_input: Dict,
        cache_output: Dict,
        unconditional: bool = False,
    ):
        """Register hook to cache activations at specified position."""
        block = self._locate_block(position)

        def hook(module, input, kwargs, output):
            if cache_input is not None:
                if position not in cache_input:
                    cache_input[position] = []
                input_to_cache = retrieve(input, unconditional)
                # Reshape to (batch, spatial_patches, d_in) if needed
                if len(input_to_cache.shape) == 4:
                    input_to_cache = input_to_cache.view(
                        input_to_cache.shape[0], input_to_cache.shape[1], -1
                    ).permute(0, 2, 1)
                cache_input[position].append(input_to_cache)

            if cache_output is not None:
                if position not in cache_output:
                    cache_output[position] = []
                output_to_cache = retrieve(output, unconditional)
                if len(output_to_cache.shape) == 4:
                    output_to_cache = output_to_cache.view(
                        output_to_cache.shape[0], output_to_cache.shape[1], -1
                    ).permute(0, 2, 1)
                cache_output[position].append(output_to_cache)

        return block.register_forward_hook(hook, with_kwargs=True)

    def _register_general_hook(self, position, hook):
        """Register general forward hook at specified position."""
        if position == "scheduler_pre":
            if not self.use_hooked_scheduler:
                raise ValueError(
                    "Cannot register hooks on scheduler without using hooked scheduler"
                )
            self.pipe.scheduler.pre_hooks.append(hook)
            return
        elif position == "scheduler_post":
            if not self.use_hooked_scheduler:
                raise ValueError(
                    "Cannot register hooks on scheduler without using hooked scheduler"
                )
            self.pipe.scheduler.post_hooks.append(hook)
            return

        block = self._locate_block(position)
        return block.register_forward_hook(hook)

    def _prepare_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        guidance_scale,
        num_inference_steps,
        generator,
        latents,
    ):
        """Prepare prompt embeddings and latents for SD 1.5."""
        # Default height and width
        height = self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        # Encode prompt (SD 1.5 has single text encoder)
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            guidance_scale > 1.0,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )

        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler, num_inference_steps, device, None, None
        )

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, 0.0)
        added_cond_kwargs = None  # SD 1.5 doesn't use added conditioning

        return prompt_embeds, timesteps, latents, extra_step_kwargs, added_cond_kwargs

    def _denoise_loop(
        self,
        timesteps,
        latents,
        guidance_scale,
        extra_step_kwargs,
        added_cond_kwargs,
        prompt_embeds,
    ):
        """Denoising loop for SD 1.5."""
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            )
            latent_model_input = self.pipe.scheduler.scale_model_input(
                latent_model_input, t
            )

            # Predict noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,  # None for SD 1.5
                return_dict=False,
            )[0]

            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Compute previous noisy sample x_t -> x_t-1
            latents = self.pipe.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

        return latents

    def _postprocess_latents(self, latents, output_type, generator):
        """Decode latents to images."""
        if not output_type == "latent":
            image = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]

        image = self.pipe.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if output_type == "latent":
            image = image.cpu().numpy()

        return image

    def to(self, *args, **kwargs):
        """Move pipeline to device."""
        self.pipe = self.pipe.to(*args, **kwargs)
        return self

    def __getattr__(self, name):
        """Delegate attribute access to underlying pipe."""
        return getattr(self.pipe, name)

    def __setattr__(self, name, value):
        """Delegate attribute setting to underlying pipe."""
        return setattr(self.pipe, name, value)

    def __call__(self, *args, **kwargs):
        """Make pipeline callable."""
        return self.pipe(*args, **kwargs)
