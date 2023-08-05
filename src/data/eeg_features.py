import numpy as np
import pywt
from scipy.signal import sosfiltfilt, butter

from src.parameters import ChannelParameters
from src.constants import N_CHANNELS


def wavelet_transform(wave: np.ndarray, channel_params: ChannelParameters,
                      cwavelet: str = 'morl') -> np.ndarray:
    scales = pywt.frequency2scale(cwavelet, channel_params.frequencies / channel_params.sample_rate)
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


def extract_features(eeg: np.ndarray, ch: int, channel_params: ChannelParameters) -> np.ndarray:
    """Get useful features from one EEG channel"""
    cleaned_signal = clean_signal(eeg[:, ch], channel_params=channel_params)
    spectrogram = wavelet_transform(cleaned_signal, channel_params=channel_params)
    spectrogram = abs_spectrogram(spectrogram, abs_mode=channel_params.abs_mode)
    return spectrogram


def extract_all_features(eeg_and_params: tuple[np.ndarray, dict[int, ChannelParameters]]) -> list[np.ndarray]:
    """Get combined spectrogram from EEG segments based on parameters"""
    eeg, params = eeg_and_params
    spectrograms = []
    for ch in range(N_CHANNELS):
        spectrogram = extract_features(eeg, ch=ch, channel_params=params[ch])
        spectrograms.append(spectrogram)
    return spectrograms


def clean_signal(signal: np.ndarray, channel_params: ChannelParameters) -> np.ndarray:
    bandpass_filter = butter(4, (channel_params.min_freq, channel_params.max_freq), 'bp',
                             output='sos', fs=channel_params.sample_rate)
    return sosfiltfilt(bandpass_filter, signal)


if __name__ == "__main__":
    from src.data.sample_gen import get_sample_eeg_segment

    eeg = get_sample_eeg_segment()
    print(f'From an EEG segment of size {eeg.shape} (samples, channels)...')
    res = extract_features(eeg, ch=0, channel_params=ChannelParameters())
    print(f'{len(res)} features produces')
    print(f'1st feature has size {res[0].shape}, min: {res[0].min()}, max: {res[0].max()}')
