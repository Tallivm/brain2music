import time
from PIL import Image

from src.data.riffusion import run_img2img


if __name__ == "__main__":

    loaded_img = Image.open('../../samples/sample_spectrogram.png')
    loaded_img = loaded_img.convert('RGB')

    start = time.time()
    res = run_img2img(
            prompt='',
            init_image=loaded_img,
            denoising_strength=0.2,
            num_inference_steps=20,
            guidance_scale=0.7,
            seed=42,
            negative_prompt=None
    )
    end = time.time()
    print(f'Completed in {end-start} s')

    start = time.time()
    res = run_img2img(
            prompt='',
            init_image=loaded_img,
            denoising_strength=0.2,
            num_inference_steps=20,
            guidance_scale=0.7,
            seed=42,
            negative_prompt=None
    )
    end = time.time()
    print(f'Completed in {end-start} s')

    res.show()
