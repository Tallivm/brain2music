import librosa as li
import numpy as np
from pandas import read_csv

from src.constants import SAMPLE_RATE, SEGMENT_LEN_S, AUDIO_SAMPLE_RATE
from src.data.utils import segment_eeg


def get_sample_eeg_segment() -> np.ndarray:
    datapath = "../../samples/eeg_samples/UnicornRecorder_20220625_121622.csv"
    data = read_csv(datapath).to_numpy()[:SAMPLE_RATE * SEGMENT_LEN_S, :8]
    return data


def get_offline_eeg_segments() -> list[np.ndarray]:
    datapath = "../../samples/eeg_samples/UnicornRecorder_20220625_121622.csv"
    data = read_csv(datapath).to_numpy()[:, :8]
    segments = segment_eeg(data, sample_rate=SAMPLE_RATE, segment_len_s=SEGMENT_LEN_S, overlap_s=0)
    return segments


def get_sample_audio_wave() -> tuple[np.ndarray, float]:
    return li.load('../../samples/sample_music.wav')


def generate_sample_wave() -> np.ndarray:
    return li.tone(261.63, sr=AUDIO_SAMPLE_RATE, duration=SEGMENT_LEN_S)
