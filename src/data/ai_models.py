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


def run_riffusion(spectrogram: np.ndarray, riffusion_model: StableDiffusionImg2ImgPipeline) -> np.ndarray:
    prepare_img = 255 - normalize_spectrogram_for_image(np.flipud(spectrogram))
    img = Image.fromarray(prepare_img).convert('RGB')
    res = run_img2img(
        pipeline=riffusion_model,
        prompt=TEXT_PROMPT,
        init_image=img,
        denoising_strength=DENOISING_STRENGTH,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        negative_prompt=TEXT_NEGATIVE_PROMPT
    )
    res_numpy = np.flipud(np.array(res.convert('L')))
    return (np.median(res_numpy) - res_numpy).clip(min=0)
