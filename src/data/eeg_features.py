import numpy as np
import pywt

from src.constants import SAMPLE_RATE, EEG_FREQUENCIES, CHANNEL_IDS


def wavelet_transform(wave: np.ndarray, frequencies: np.ndarray, sample_rate: int,
                      cwavelet: str = 'morl') -> np.ndarray:
    scales = pywt.frequency2scale(cwavelet, frequencies / sample_rate)
    transformed = pywt.cwt(wave, scales=scales, wavelet=cwavelet)[0]
    return transformed


def abs_spectrogram(spectrogram: np.ndarray, abs_mode: str) -> np.ndarray:
    if abs_mode == 'none':
        return spectrogram
    elif abs_mode == 'both':
        return np.abs(spectrogram)
    elif abs_mode == 'plus':
        return spectrogram.clip(0, spectrogram.max())
    elif abs_mode == 'minus':
        return np.abs(spectrogram.clip(spectrogram.min(), 0))
    else:
        raise ValueError('abs_mode can only be "none", "both", "plus" or "minus"')


def extract_features(eeg: np.ndarray, frequencies: np.ndarray = EEG_FREQUENCIES, channels: list[int] = CHANNEL_IDS,
                     abs_mode: str = 'both') -> list[np.ndarray]:
    """Get chosen channels and some other useful features from EEG data"""
    spectrograms = []
    for ch in channels:
        spectrogram = wavelet_transform(eeg[:, ch], frequencies=frequencies, sample_rate=SAMPLE_RATE)
        spectrogram = abs_spectrogram(spectrogram, abs_mode=abs_mode)
        spectrograms.append(spectrogram)
    return spectrograms


if __name__ == "__main__":
    from src.data.utils import get_sample_eeg_segment
    eeg = get_sample_eeg_segment()
    print(f'From an EEG segment of size {eeg.shape} (samples, channels)...')
    res = extract_features(eeg)
    print(f'{len(res)} features produces')
    print(f'1st feature has size {res[0].shape}, min: {res[0].min()}, max: {res[0].max()}')
