import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

from src.data.utils import normalize_spectrogram_for_image
from src.data.riffusion import run_img2img
from src.constants import TEXT_PROMPT, TEXT_NEGATIVE_PROMPT, DENOISING_STRENGTH, GUIDANCE_SCALE, INFERENCE_STEPS


def load_rave_model(model_name: str):
    torch.set_grad_enabled(False)
    model = torch.jit.load(f"../../models/{model_name}.ts").eval()
    return model


def run_rave(wave: np.ndarray, rave_model) -> np.ndarray:
    x = torch.from_numpy(wave).reshape(1, 1, -1)
    z = rave_model.encode(x)
    z[:, 0] += torch.linspace(-2, 2, z.shape[-1])  # what happens there?
    y = rave_model.decode(z)
    y = y.numpy().reshape(-1)
    return y


def run_riffusion(spectrogram: np.ndarray, riffusion_model: StableDiffusionImg2ImgPipeline,
                  generator: torch.Generator) -> np.ndarray:
    img = Image.fromarray(normalize_spectrogram_for_image(spectrogram)).convert('RGB')
    res = run_img2img(
        pipeline=riffusion_model,
        prompt=TEXT_PROMPT,
        init_image=img,
        denoising_strength=DENOISING_STRENGTH,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        negative_prompt=TEXT_NEGATIVE_PROMPT
    )
    res_numpy = np.array(res.convert('L'))
    return 255 - normalize_spectrogram_for_image(res_numpy.clip(0, np.median(res_numpy)))
