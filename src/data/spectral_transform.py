import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
import skimage

from src.data.utils import resize_image, normalize_spectrogram_with_max_power, normalize_spectrogram
from src.data.sample_gen import generate_sample_wave
from src.constants import SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, SPECTROGRAM_SHIFT, NOTE_MASK, MEL_NOTES
from src.data.ai_models import run_rave, run_riffusion

from typing import Optional


def combine_spectrograms(spectrograms: list[np.ndarray]) -> np.ndarray:
    """Build a spectrogram out of collected EEG features"""
    spectrogram = np.mean(np.array(spectrograms), axis=0)
    spectrogram = resize_image(spectrogram, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)
    # TODO: think of a better building instead of just channel averaging
    return spectrogram


def transform_spectrogram(spectrogram: np.ndarray,
                          riffusion_model: Optional[StableDiffusionImg2ImgPipeline] = None,
                          measure_difference: bool = True) -> np.ndarray:
    """Apply algorithms over the whole spectrogram"""
    transformed = np.roll(spectrogram, shift=SPECTROGRAM_SHIFT, axis=0)
    transformed = filter_spectrogram(transformed)
    if riffusion_model is not None:
        transformed = run_riffusion(spectrogram=transformed, riffusion_model=riffusion_model)
    transformed = normalize_spectrogram_with_max_power(transformed, with_power=True)
    if measure_difference:
        diff = measure_diff_between_spectrograms(spectrogram, transformed)
        print(f'Difference metrics | mean={diff[0]}, median={diff[1]}, rmse={diff[2]}')
    return transformed


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
    blurred[MEL_NOTES] = blurred[MEL_NOTES] ** 2
    return blurred


def measure_diff_between_spectrograms(spectra0: np.ndarray, spectra1: np.ndarray) -> tuple[float, float, float]:
    norm_diff = np.abs(normalize_spectrogram(spectra0) - normalize_spectrogram(spectra1))
    norm_mean_diff = np.mean(norm_diff)
    norm_median_diff = np.median(norm_diff)
    rmse = np.sqrt(np.mean(norm_diff ** 2))
    return norm_mean_diff, norm_median_diff, rmse
