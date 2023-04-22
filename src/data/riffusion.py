import threading
from dataclasses import dataclass
from typing import Optional, Any, Callable

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

from src.constants import RIFFUSION_CHECKPOINT, SCHEDULER_OPTIONS


@dataclass(frozen=True)
class PromptInput:
    """Parameters for one end of interpolation."""
    # Text prompt fed into a CLIP model
    prompt: str
    # Random seed for denoising
    seed: int
    # Negative prompt to avoid (optional)
    negative_prompt: Optional[str] = None
    # Denoising strength
    denoising: float = 0.75
    # Classifier-free guidance strength
    guidance: float = 7.0


@dataclass(frozen=True)
class InferenceInput:
    """
    Parameters for a single run of the riffusion model, interpolating between
    a start and end set of PromptInputs. This is the API required for a request
    to the model server.
    """
    # Start point of interpolation
    start: PromptInput
    # End point of interpolation
    end: PromptInput
    # Interpolation alpha [0, 1]. A value of 0 uses start fully, a value of 1 uses end fully.
    alpha: float
    # Number of inner loops of the diffusion model
    num_inference_steps: int = 50
    # Which seed image to use
    seed_image_id: str = "og_beat"
    # ID of mask image to use
    mask_image_id: Optional[str] = None


def pipeline_lock() -> threading.Lock:
    """Singleton lock used to prevent concurrent access to any model pipeline."""
    return threading.Lock()


def get_scheduler(scheduler: str, config: Any) -> Any:
    """
    Construct a denoising scheduler from a string.
    """
    if scheduler == "PNDMScheduler":
        from diffusers import PNDMScheduler

        return PNDMScheduler.from_config(config)
    elif scheduler == "DPMSolverMultistepScheduler":  # <--- Usually this one is used
        from diffusers import DPMSolverMultistepScheduler

        return DPMSolverMultistepScheduler.from_config(config)
    elif scheduler == "DDIMScheduler":
        from diffusers import DDIMScheduler

        return DDIMScheduler.from_config(config)
    elif scheduler == "LMSDiscreteScheduler":
        from diffusers import LMSDiscreteScheduler

        return LMSDiscreteScheduler.from_config(config)
    elif scheduler == "EulerDiscreteScheduler":
        from diffusers import EulerDiscreteScheduler

        return EulerDiscreteScheduler.from_config(config)
    elif scheduler == "EulerAncestralDiscreteScheduler":
        from diffusers import EulerAncestralDiscreteScheduler

        return EulerAncestralDiscreteScheduler.from_config(config)
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")


def load_stable_diffusion_img2img_pipeline(
    checkpoint: str = RIFFUSION_CHECKPOINT,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    scheduler: str = SCHEDULER_OPTIONS[0],
) -> StableDiffusionImg2ImgPipeline:
    """
    Load the image to image pipeline.
    TODO(hayk): Merge this into RiffusionPipeline to just load one model.
    """
    if device == "cpu" or device.lower().startswith("mps"):
        print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
        dtype = torch.float32

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        checkpoint,
        revision="main",
        torch_dtype=dtype,
        safety_checker=lambda images, **kwargs: (images, False),
    ).to(device)
    pipeline.scheduler = get_scheduler(scheduler, config=pipeline.scheduler.config)
    return pipeline


def run_img2img(
    prompt: str,
    init_image: Image.Image,
    denoising_strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    negative_prompt: Optional[str] = None,
    checkpoint: str = RIFFUSION_CHECKPOINT,
    device: str = "cuda",
    scheduler: str = SCHEDULER_OPTIONS[0],
    progress_callback: Optional[Callable[[float], Any]] = None,
) -> Image.Image:
    with pipeline_lock():
        pipeline = load_stable_diffusion_img2img_pipeline(
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )

        generator_device = "cpu" if device.lower().startswith("mps") else device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        num_expected_steps = max(int(num_inference_steps * denoising_strength), 1)

        def callback(step: int, tensor: torch.Tensor, foo: Any) -> None:
            if progress_callback is not None:
                progress_callback(step / num_expected_steps)

        result = pipeline(
            prompt=prompt,
            image=init_image,
            strength=denoising_strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt or None,
            num_images_per_prompt=1,
            generator=generator,
            callback=callback,
            callback_steps=1,
        )

        return result.images[0]
