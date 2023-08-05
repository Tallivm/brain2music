import os
import re
from io import BytesIO
import numpy as np
import pydub
from pydub import AudioSegment
import librosa as li
from scipy.io import wavfile
import skimage

from src.constants import AUDIO_SAMPLE_RATE, SPECTROGRAM_MAX_VALUE, SPECTROGRAM_POWER, DESIRED_DB, CROSSFADE_SAVE_MS, \
    DEFAULT_SAVE_AUDIO_FOLDER, ANTISPIKE_THRESHOLD


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


def postprocess_wave(wave: np.ndarray, threshold: float = ANTISPIKE_THRESHOLD) -> np.ndarray:
    # wave = np.where(np.abs(wave) > threshold, np.sign(wave) * threshold, wave)
    return wave


def apply_audio_filters(audio: AudioSegment) -> AudioSegment:
    # audio = pydub.effects.normalize(audio, headroom=0.1)
    # audio = audio.apply_gain((DESIRED_DB + 2) - audio.dBFS)
    # audio = pydub.effects.compress_dynamic_range(audio, threshold=-25.0, ratio=3.0, attack=10.0, release=100)
    audio = audio.apply_gain(DESIRED_DB - audio.dBFS)
    audio = pydub.effects.normalize(audio, headroom=0.1)
    return audio


def save_pydub_audio_file(audio: AudioSegment, filepath: str) -> None:
    audio.export(filepath, format="wav")


def save_audio_to_file(numbered_audio: tuple[int, AudioSegment],
                       save_folder: str = DEFAULT_SAVE_AUDIO_FOLDER) -> AudioSegment:
    """To use in Stream, saves produced audio with a respective number in name"""
    number, audio = numbered_audio
    audio.export(f'../../samples/outputs/{save_folder}/{number:04}.wav', 'wav')
    return audio


def combine_pydub_audio_from_queue(queue) -> AudioSegment:  # queue: torch.Queue
    combined = queue.get()
    while not queue.empty():
        combined = combined.append(queue.get(), crossfade=CROSSFADE_SAVE_MS)
    return combined


def combine_pydub_audio_from_folder(folder_path: str) -> AudioSegment:
    audio_files = sorted(os.listdir(folder_path))
    assert len(audio_files) > 1, "No meaning in combining less than 2 files"
    combined = AudioSegment.from_wav(os.path.join(folder_path, audio_files[0]))
    for f in audio_files[1:]:
        segment = AudioSegment.from_wav(os.path.join(folder_path, f))
        combined = combined.append(segment, crossfade=CROSSFADE_SAVE_MS)
    return combined


def normalize_spectrogram(s: np.ndarray) -> np.ndarray:
    return (s - s.min()) / (s.max() - s.min())


def invert_normalized_spectrogram(s: np.ndarray) -> np.ndarray:
    return 1 - s


def normalize_spectrogram_with_max_power(s: np.ndarray, with_power: bool = False) -> np.ndarray:
    normalized = normalize_spectrogram(s) * SPECTROGRAM_MAX_VALUE
    if with_power:
        normalized = np.power(normalized, SPECTROGRAM_POWER)
    return normalized


def normalize_spectrogram_for_image(s: np.ndarray) -> np.ndarray:
    return (normalize_spectrogram(s) * 255).astype('uint8')


def save_spectrogram_as_image(s: np.ndarray, filepath: str, inverse: bool = False, flip: bool = False,
                              as_rgb: bool = False) -> None:
    s_img = normalize_spectrogram_for_image(s)
    if inverse:
        s_img = 255 - s_img
    if flip:
        s_img = np.flipud(s_img)
    if as_rgb:
        s_img = np.tile(np.expand_dims(s_img, axis=-1), 3)
    skimage.io.imsave(filepath, s_img)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]
