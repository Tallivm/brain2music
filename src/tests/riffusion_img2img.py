import time
import numpy as np
from PIL import Image, ImageOps
import skimage

from src.data.riffusion import load_stable_diffusion_img2img_pipeline, get_generator, run_img2img
from src.data.utils import produce_audio_from_spectrogram_with_torch, save_pydub_audio_file
from src.data.utils import normalize_spectrogram_with_max_power, normalize_spectrogram_for_image
from src.data.torch_utils import SpectrogramConverter, check_device
from src.constants import TEXT_PROMPT, TEXT_NEGATIVE_PROMPT, GUIDANCE_SCALE, DENOISING_STRENGTH, INFERENCE_STEPS


if __name__ == "__main__":

    loaded_img = Image.open('../../samples/eeg_sample_to_audio.png')
    loaded_img = ImageOps.invert(loaded_img).convert('RGB')

    device = check_device('cuda')
    riffusion_pipeline = load_stable_diffusion_img2img_pipeline()
    generator = get_generator(42, device)

    start = time.time()
    res = run_img2img(
        pipeline=riffusion_pipeline,
        prompt=TEXT_PROMPT,
        init_image=loaded_img,
        denoising_strength=DENOISING_STRENGTH,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        negative_prompt=TEXT_NEGATIVE_PROMPT
    )
    end = time.time()
    print(f'Completed in {end-start} s')

    res_numpy = np.array(res.convert('L'))
    res_numpy = 255 - normalize_spectrogram_for_image(res_numpy.clip(0, np.median(res_numpy)))
    skimage.io.imsave('../../samples/riffusion_result.png', res_numpy)

    converter = SpectrogramConverter()
    res_numpy = normalize_spectrogram_with_max_power(res_numpy)
    audio = produce_audio_from_spectrogram_with_torch(res_numpy, converter)
    save_pydub_audio_file(audio, '../../samples/riffusion_result.wav')
