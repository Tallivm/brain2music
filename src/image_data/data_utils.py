import math
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
import pywt

from typing import Any

from src.image_data.image_utils import resize_image
from src.constants import VERTICAL_RESOLUTION, HORIZONTAL_RESOLUTION, SAMPLE_EEG_FILEPATH, SAMPLE_RATE, SEGMENT_LEN_S, \
    FREQUENCIES


def load_eeg_mat_file(filepath: str) -> dict[int, dict[str, Any]]:
    raw_data = loadmat(filepath)['data'][0]
    records = {}
    for i, record in enumerate(raw_data):
        records[i] = {}
        keys = record.dtype.names
        for key in keys:
            value = np.squeeze(record[key].item())
            records[i][key] = value
    return records


def resample_eeg(eeg: np.ndarray, old_sample_rate: int, new_sample_rate: int) -> np.ndarray:
    """Data should be of shape: (signal, channels)"""
    num = round(eeg.shape[0] * (new_sample_rate / old_sample_rate))
    resampled = signal.resample(eeg, num, domain='time')
    return resampled


def cut_array(arr: np.ndarray, max_range: int, range_step: int) -> list[np.ndarray]:
    samples = []
    cut_points = np.arange(0, max_range, range_step)
    for i in range(len(cut_points) - 1):
        sample = arr[cut_points[i]: cut_points[i+1]]
        samples.append(sample)
    return samples


def segment_eeg(eeg: np.ndarray, sample_rate: int, segment_len_s: int, overlap_s: int = 0) -> list[np.ndarray]:
    """Data should be of shape: (signal, channels)"""
    # TODO: check if works correctly, especially the step parameter
    eeg_len = eeg.shape[0]
    sample_len = sample_rate * segment_len_s
    overlap_len = overlap_s * sample_rate
    samples = cut_array(eeg, eeg_len - sample_len, sample_len - overlap_len)
    return samples


def wavelet_transform(channel: np.ndarray, freqs: np.ndarray, sample_rate: int, cwavelet: str = 'morl') -> np.ndarray:
    scales = pywt.frequency2scale(cwavelet, freqs / sample_rate)
    transformed = pywt.cwt(channel, scales=scales, wavelet=cwavelet)[0]
    return transformed


def spectrogram_as_image(spectrogram: np.ndarray, to_uint: bool = False) -> np.ndarray:
    res = ((spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min()))
    res = res.clip(0, 1)
    if to_uint:
        res = 255 - (res * 255).astype('uint8')
    return res


def spectrogram_for_converter(spectrogram: np.ndarray, power: float = 0.25,
                              max_power: float = 30e6) -> np.ndarray:
    data = spectrogram_as_image(spectrogram, to_uint=True)
    # data = np.flipud(data)
    data = np.expand_dims(data, 0).astype(np.float32)
    # data = 255 - data
    # data = data / 255
    data = np.power(data, 1 / power)
    return data * max_power


def combine_spectrograms(spectrograms: list[np.ndarray]) -> np.ndarray:
    return np.mean(spectrograms, axis=0)


def eeg2spectrogram(eeg: np.ndarray, freqs: np.ndarray, fs: int, to_abs: bool = True) -> np.ndarray:
    spectrograms = []
    for ch in range(eeg.shape[1]):
        spectrogram = wavelet_transform(channel=eeg[:, ch], freqs=freqs, sample_rate=fs,
                                        cwavelet='morl')
        if to_abs:
            spectrogram = np.abs(spectrogram)
        spectrograms.append(spectrogram)
    joined_spectrogram = combine_spectrograms(spectrograms)
    joined_spectrogram = spectrogram_as_image(joined_spectrogram, to_uint=True)
    joined_spectrogram = resize_image(joined_spectrogram, HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION)
    return joined_spectrogram


def hz2mel(x: float) -> float:
    return 2595 * np.log10(1 + x / 700)


def generate_sinewave(x: np.ndarray, freq: float, fs: int, amplitude: float) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * x / fs)


def get_sample_spectrogram() -> np.ndarray:
    segment = pd.read_csv(SAMPLE_EEG_FILEPATH).to_numpy()[:SEGMENT_LEN_S * SAMPLE_RATE, :1]
    spectrogram = eeg2spectrogram(segment, FREQUENCIES, SAMPLE_RATE, to_abs=True)
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
