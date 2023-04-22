import numpy as np

from src.data.utils import resize_image
from src.data.sample_gen import generate_sample_wave
from src.constants import SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT
from src.data.ai_models import run_rave


def build_spectrogram_from_eeg_features(eeg_features: list[np.ndarray]) -> np.ndarray:
    """Build a spectrogram out of collected EEG features"""
    spectrogram = np.mean(np.array(eeg_features), axis=0)
    spectrogram = resize_image(spectrogram, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)
    # TODO: think of a better building instead of just channel averaging
    return spectrogram


def transform_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """Apply algorithms over the whole spectrogram"""
    return spectrogram


def transform_wave(wave: np.ndarray, rave_model, add_background_sound: bool = False) -> np.ndarray:
    """Apply algorithms over the whole wave"""
    transformed = run_rave(wave, rave_model)
    if add_background_sound:
        to_add = generate_sample_wave()
        transformed = transformed[:len(to_add)] + to_add
    return transformed
