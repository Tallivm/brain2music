import threading
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

from src.constants import RIFFUSION_CHECKPOINT, SCHEDULER_OPTIONS, SEED

from typing import Optional, Any, Callable


def pipeline_lock() -> threading.Lock:
    """Singleton lock used to prevent concurrent access to any model pipeline."""
    return threading.Lock()


def get_scheduler(scheduler: str, config: Any) -> Any:
    """Construct a denoising scheduler from a string."""
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


def nsfw_disabler(images, **kwargs):
    return images, False


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
        safety_checker=nsfw_disabler,
    ).to(device)
    pipeline.scheduler = get_scheduler(scheduler, config=pipeline.scheduler.config)
    return pipeline


def get_generator(seed: int, device: str) -> torch.Generator:
    generator_device = "cpu" if device.lower().startswith("mps") else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    return generator


def run_img2img(
    pipeline: StableDiffusionImg2ImgPipeline,
    prompt: str,
    init_image: Image.Image,
    denoising_strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: Optional[str] = None,
    progress_callback: Optional[Callable[[float], Any]] = None,
    device: str = 'cuda'
) -> Image.Image:
    def callback(step: int, tensor: torch.Tensor, foo: Any,) -> None:
        num_expected_steps = max(int(num_inference_steps * denoising_strength), 1)
        if progress_callback is not None:
            progress_callback(step / num_expected_steps)

    generator = get_generator(SEED, device)
    with pipeline_lock():
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
