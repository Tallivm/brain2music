import io
import numpy as np
import pydub
from scipy.io import wavfile


def audio_from_waveform(samples: np.ndarray, sample_rate: int, normalize: bool = False) -> pydub.AudioSegment:
    if normalize:
        samples *= np.iinfo(np.int16).max / np.max(np.abs(samples))
    # samples = samples.transpose(1, 0)  # not needed anymore?
    samples = samples.astype(np.int16)
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples)
    wav_bytes.seek(0)
    return pydub.AudioSegment.from_wav(wav_bytes)
