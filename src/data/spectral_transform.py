import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
import skimage

from src.data.utils import resize_image, normalize_spectrogram_with_max_power
from src.data.sample_gen import generate_sample_wave
from src.constants import SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, NOTE_MASK
from src.data.ai_models import run_rave, run_riffusion

from typing import Optional


def build_spectrogram_from_eeg_features(eeg_features: list[np.ndarray]) -> np.ndarray:
    """Build a spectrogram out of collected EEG features"""
    spectrogram = np.mean(np.array(eeg_features), axis=0)
    spectrogram = resize_image(spectrogram, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)
    # TODO: think of a better building instead of just channel averaging
    return spectrogram


def transform_spectrogram(spectrogram: np.ndarray,
                          riffusion_model: Optional[StableDiffusionImg2ImgPipeline] = None) -> np.ndarray:
    """Apply algorithms over the whole spectrogram"""
    transformed = spectrogram.copy()
    transformed = filter_spectrogram(transformed)
    if riffusion_model is not None:
        transformed = run_riffusion(spectrogram=transformed, riffusion_model=riffusion_model)
    return normalize_spectrogram_with_max_power(transformed)


def transform_wave(
        wave: np.ndarray,
        rave_model: Optional = None,
        add_background_sound: bool = False
) -> np.ndarray:
    """Apply algorithms over the whole wave"""
    transformed = wave.copy()
    if rave_model is not None:
        transformed = run_rave(wave, rave_model)
    if add_background_sound:
        to_add = generate_sample_wave()
        transformed = transformed[:len(to_add)] + to_add
    return transformed


def filter_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    thresholded = spectrogram > skimage.filters.threshold_triangle(spectrogram)
    tuned = thresholded * NOTE_MASK
    blurred = skimage.filters.gaussian(tuned, sigma=1, mode='constant')
    return blurred
