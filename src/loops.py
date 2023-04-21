import time
from multiprocessing import Queue

import numpy as np

from src.image_data.data_utils import eeg2spectrogram
from src.audio_data.audio_utils import play_audio_from_buffer


def eeg2img(eeg_queue: Queue, img_queue: Queue, freq: np.ndarray, fs: int) -> None:
    eeg_data = eeg_queue.get()
    spectrogram = eeg2spectrogram(eeg_data, freq, fs)
    img_queue.put(spectrogram)
    print('Spectrogram produced')


def eeg2img_loop(eeg_queue: Queue, img_queue: Queue, freq: np.ndarray, fs: int) -> None:
    while True:
        if not eeg_queue.empty():
            eeg2img(eeg_queue, img_queue, freq, fs)
        time.sleep(0.5)


def img2audio(img_queue: Queue, audio_queue: Queue, converter) -> None:
    img = img_queue.get()
    audio = converter.audio_from_spectrogram(img)
    audio_queue.put(audio)
    print('Audio segment produced')


def img2audio_loop(img_queue: Queue, audio_queue: Queue, converter) -> None:
    while True:
        if not img_queue.empty():
            img2audio(img_queue, audio_queue, converter)
        time.sleep(0.5)


def player_loop(audio_queue: Queue) -> None:
    while True:
        if not audio_queue.empty():
            audio = audio_queue.get()
            print('Will play audio_data now')
            play_audio_from_buffer(audio)


