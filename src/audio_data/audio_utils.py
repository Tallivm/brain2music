import io
import time

import numpy as np
import simpleaudio
from pydub import AudioSegment
from scipy.io import wavfile


def audio_from_waveform(samples: np.ndarray, sample_rate: int, normalize: bool = False) -> AudioSegment:
    if normalize:
        samples *= np.iinfo(np.int16).max / np.max(np.abs(samples))
    samples = samples.squeeze()
    # samples = np.expand_dims(samples, -1)
    samples = samples.astype(np.int16)
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples)
    wav_bytes.seek(0)
    return AudioSegment.from_wav(wav_bytes)


def spectrogram2audio(spectrogram: np.ndarray, converter) -> AudioSegment:
    return converter.audio_from_spectrogram(spectrogram)


def play_audio(audio: AudioSegment, sleep_time: float = 5.09) -> None:
    print('playing a sound')
    simpleaudio.play_buffer(audio.raw_data,
                            num_channels=audio.channels,
                            bytes_per_sample=audio.sample_width,
                            sample_rate=audio.frame_rate
                            )
    time.sleep(sleep_time)
