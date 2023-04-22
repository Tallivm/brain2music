from io import BytesIO
import numpy as np
from pydub import AudioSegment
import librosa as li
from scipy.io import wavfile
import skimage

from src.constants import AUDIO_SAMPLE_RATE, SPECTROGRAM_MAX_VALUE


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


def resize_image(img: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    height = img.shape[0] if height is None else height
    width = img.shape[1] if width is None else width
    return skimage.transform.resize(img, (height, width), anti_aliasing=False)


def produce_audio_from_spectrogram_with_librosa(spectrogram: np.ndarray) -> AudioSegment:
    wave = li.griffinlim(spectrogram).astype(np.int16)
    return produce_audio_from_wave(wave)


def produce_audio_from_spectrogram_with_torch(spectrogram: np.ndarray, converter) -> AudioSegment:
    return converter.audio_from_spectrogram(spectrogram)


def produce_wave_with_torch(spectrogram: np.ndarray, converter) -> np.ndarray:
    return converter.wave_from_spectrogram(spectrogram)


def produce_audio_from_wave(wave: np.ndarray, normalize: bool = True) -> AudioSegment:
    if normalize:
        wave = wave * np.iinfo(np.int16).max / np.max(np.abs(wave))
    wav_bytes = BytesIO()
    wavfile.write(wav_bytes, rate=AUDIO_SAMPLE_RATE, data=wave.astype(np.int16))
    wav_bytes.seek(0)
    return AudioSegment.from_wav(wav_bytes)


def save_pydub_audio_file(audio: AudioSegment, filepath: str) -> None:
    audio.export(filepath, format="wav")


def normalize_spectrogram_with_max_power(s: np.ndarray) -> np.ndarray:
    return (s - s.min()) / (s.max() - s.min()) * SPECTROGRAM_MAX_VALUE


def normalize_spectrogram_for_image(s: np.ndarray) -> np.ndarray:
    return ((s - s.min()) / (s.max() - s.min()) * 255).astype('uint8')


def save_spectrogram_as_image(s: np.ndarray, filepath: str) -> None:
    s_img = normalize_spectrogram_for_image(s)
    skimage.io.imsave(filepath, s_img)
