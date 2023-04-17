import numpy as np
from scipy import signal
from scipy.io import loadmat
import pywt
import skimage
from pydub import AudioSegment
from UnicornPy import SamplingRate

from custom_riffusion import SpectrogramConverter

from typing import Any


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


def resample_data(eeg: np.ndarray, old_sample_rate: int, new_sample_rate: int) -> np.ndarray:
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
    if to_uint:
        res = 255 - (res * 255).astype('uint8')
    res = np.flip(res)
    return res


def resize_image(img: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    height = img.shape[0] if height is None else height
    width = img.shape[1] if width is None else width
    return skimage.transform.resize(img, (height, width), anti_aliasing=False)


def combine_spectrograms(spectrograms: list[np.ndarray]) -> np.ndarray:
    return np.mean(spectrograms, axis=0)


def eeg2spectrogram(eeg: np.ndarray) -> np.ndarray:
    spectrograms = []
    for ch in range(eeg.shape[1]):
        spectrogram = wavelet_transform(channel=eeg[:, ch], freqs=np.arange(1, 30), sample_rate=SamplingRate,
                                        cwavelet='morl')
        spectrograms.append(spectrogram)
    joined_spectrogram = combine_spectrograms(spectrograms)
    joined_spectrogram = spectrogram_as_image(joined_spectrogram, to_uint=True)
    joined_spectrogram = resize_image(joined_spectrogram, 512, 512)
    return joined_spectrogram


def spectrogram2audio(spectrogram: np.ndarray, converter: SpectrogramConverter) -> AudioSegment:
    return converter.audio_from_spectrogram(spectrogram)
