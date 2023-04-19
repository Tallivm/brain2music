import time
from multiprocessing import Queue

import numpy as np

from src.image_data.data_utils import eeg2spectrogram
from src.audio_data.audio_utils import play_audio, spectrogram2audio


def eeg2img_loop(eeg_queue: Queue, img_queue: Queue, freq: np.ndarray, fs: int) -> None:
    while True:
        if not eeg_queue.empty():
            eeg_data = eeg_queue.get()
            spectrogram = eeg2spectrogram(eeg_data, freq, fs)
            img_queue.put(spectrogram)
            print('Spectrogram produced')
        time.sleep(0.5)


def img2audio_loop(img_queue: Queue, audio_queue: Queue, converter) -> None:
    while True:
        if not img_queue.empty():
            img = img_queue.get()
            audio = spectrogram2audio(img, converter)
            audio_queue.put(audio)
            print('Audio segment produced')
        time.sleep(0.5)


def player(audio_queue: Queue) -> None:
    while True:
        if not audio_queue.empty():
            audio = audio_queue.get()
            print('Will play audio_data now')
            play_audio(audio)


def acquire_eeg(queue: Queue, record_size: int, fs: int) -> None:
    from src.unicorn.unicorn_utils import connect_to_unicorn, calculate_buffer_len, acquire_eeg_data_record, reshape_eeg_record
    print(f'Connecting to Unicorn...')
    device = connect_to_unicorn()
    n_scans = record_size * fs
    n_channels = device.GetNumberOfAcquiredChannels()
    buffer_len = calculate_buffer_len(n_scans, n_channels, buffer_size=4)
    print(f'Buffer length will be: {buffer_len}')
    buffer = bytearray(buffer_len)

    device.StartAcquisition(False)

    while True:
        try:
            data = acquire_eeg_data_record(device, n_scans, buffer, buffer_len, n_channels)
            data = reshape_eeg_record(data, n_scans, n_channels)
            queue.put(data)
            print('Collected data sample')
        except Exception as e:
            print(e)
            device.StopAcquisition()
            break
