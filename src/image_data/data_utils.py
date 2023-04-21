import math
import numpy as np
import pandas as pd
import pywt
from PIL import Image

from src.image_data.image_utils import resize_image
from src.constants import VERTICAL_RESOLUTION, HORIZONTAL_RESOLUTION, SAMPLE_EEG_FILEPATH, SAMPLE_RATE, SEGMENT_LEN_S, \
    FREQUENCIES, RIFFUSION_MAX_POWER


def cut_array(arr: np.ndarray, segment_len: int, overlap_len: int) -> list[np.ndarray]:
    samples = []
    arr_len = arr.shape[0]
    for i in range(0, arr_len - segment_len + 1, segment_len - overlap_len):
        sample = arr[i:i + segment_len]
        samples.append(sample)

    return samples


def segment_eeg(eeg: np.ndarray, sample_rate: int, segment_len_s: int, overlap_s: int = 0) -> list[np.ndarray]:
    assert eeg.ndim == 2, "eeg must be a 2-dimensional array of shape (signal, channels)"
    assert segment_len_s > 0, "segment_len_s must be a positive number"
    assert overlap_s >= 0, "overlap_s must be a non-negative number"

    sample_len = round(sample_rate * segment_len_s)
    overlap_len = round(overlap_s * sample_rate)
    samples = cut_array(eeg, sample_len, overlap_len)
    return samples


def wavelet_transform(channel: np.ndarray, freqs: np.ndarray, sample_rate: int, cwavelet: str = 'morl') -> np.ndarray:
    scales = pywt.frequency2scale(cwavelet, freqs / sample_rate)
    transformed = pywt.cwt(channel, scales=scales, wavelet=cwavelet)[0]
    return transformed


def normalize_img(img: np.ndarray) -> np.ndarray:
    res = (img - img.min()) / (img.max() - img.min())
    return res.clip(0, 1)


def img_float2uint(img: np.ndarray) -> np.ndarray:
    return (img * 255).astype('uint8')


def spectrum2riff_spectrum(spectrogram: np.ndarray, power: float = 0.25,
                           max_power: float = RIFFUSION_MAX_POWER) -> np.ndarray:
    """Convert an image to riffusion spectrogram"""
    data = normalize_img(spectrogram)
    data = np.expand_dims(data, 0).astype(np.float32)
    data = np.power(data, 1 / power)
    return data * max_power


def pil_image2img(image: Image.Image, inversed: bool = False) -> np.ndarray:
    """Used to convert images loaded with PIL to uint8 numpy arrays"""
    if image.mode in ("P", "L"):
        image = image.convert("RGB")
    data = np.array(image).transpose(2, 0, 1)
    if inversed:
        data = 255 - data
    return data[0, :, :]


def combine_spectrograms(spectrograms: list[np.ndarray]) -> np.ndarray:
    return np.mean(spectrograms, axis=0)


def abs_spectrogram(spectrogram: np.ndarray, abs_mode: str) -> np.ndarray:
    if abs_mode == 'none':
        return spectrogram
    elif abs_mode == 'both':
        return np.abs(spectrogram)
    elif abs_mode == 'plus':
        return spectrogram.clip(0, spectrogram.max())
    elif abs_mode == 'minus':
        return spectrogram.clip(spectrogram.min(), 0)
    else:
        raise ValueError('abs_mode can only be "none", "both", "plus" or "minus"')


def eeg2spectrogram(eeg: np.ndarray, freqs: np.ndarray, fs: int, abs_mode: str = 'none') -> np.ndarray:
    spectrograms = []
    for ch in range(eeg.shape[1]):
        spectrogram = wavelet_transform(channel=eeg[:, ch], freqs=freqs, sample_rate=fs, cwavelet='morl')
        spectrogram = abs_spectrogram(spectrogram, abs_mode)
        spectrograms.append(spectrogram)
    joined_spectrogram = combine_spectrograms(spectrograms)
    joined_spectrogram = resize_image(joined_spectrogram, HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION)
    return joined_spectrogram


def hz2mel(x: float) -> float:
    return 2595 * np.log10(1 + x / 700)


def generate_sinewave(x: np.ndarray, freq: float, fs: int, amplitude: float) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * x / fs)


def get_sample_spectrogram(abs_mode: str = 'both') -> np.ndarray:
    segment = pd.read_csv(SAMPLE_EEG_FILEPATH).to_numpy()[:SEGMENT_LEN_S * SAMPLE_RATE, :1]
    spectrogram = eeg2spectrogram(segment, FREQUENCIES, SAMPLE_RATE, abs_mode=abs_mode)
    return spectrogram


def find_nearest_pos(arr: np.array, values: list[float]) -> list[int]:
    idx = np.searchsorted(arr, values, side="left")
    pos = []
    for value, i in zip(values, idx):
        if i > 0 and (i == len(arr) or math.fabs(value - arr[i-1]) < math.fabs(value - arr[i])):
            pos.append(i - 1)
        else:
            pos.append(i)
    return pos


def get_note_positions(note_freqs: list[float], min_freq: float, max_freq: float) -> list[int]:
    freqs = np.linspace(min_freq, max_freq, VERTICAL_RESOLUTION)
    pos = find_nearest_pos(freqs, note_freqs)
    return pos


def leave_notes_in_spectrogram(spectrogram: np.ndarray, note_freqs: list[float],
                               min_freq: float, max_freq: float) -> np.ndarray:
    note_positions = get_note_positions(note_freqs, min_freq, max_freq)
    mask = np.ones((VERTICAL_RESOLUTION, HORIZONTAL_RESOLUTION))
    mask[note_positions] = spectrogram[note_positions]
    return mask
