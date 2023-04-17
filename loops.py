import time
from multiprocessing import Queue

import UnicornPy

from data_utils import eeg2spectrogram, spectrogram2audio
from audio_utils import play_audio
from unicorn_utils import connect_to_unicorn, calculate_buffer_len, acquire_eeg_data_record, reshape_eeg_record
from custom_riffusion import SpectrogramConverter


def transformer(eeg_queue: Queue, audio_queue: Queue, converter: SpectrogramConverter) -> None:
    while True:
        if not eeg_queue.empty():
            eeg_data = eeg_queue.get()
            spectrogram = eeg2spectrogram(eeg_data)
            audio = spectrogram2audio(spectrogram, converter)
            audio_queue.put(audio)
            print('Audio segment produced')
        time.sleep(0.5)


def player(audio_queue: Queue) -> None:
    while True:
        if not audio_queue.empty():
            audio = audio_queue.get()
            print('Will play audio now')
            play_audio(audio)


def acquire_eeg(queue: Queue, record_size: int, channels_to_acquire: list[int]) -> None:
    print(f'Connecting to Unicorn...')
    device = connect_to_unicorn()
    n_scans = record_size * UnicornPy.SamplingRate

    # channels = device.GetConfiguration().Channels
    # new_config = UnicornPy.AmplifierConfiguration()
    # new_config.Channels = [c for i, c in enumerate(channels) if i in channels_to_acquire]
    # device.SetConfiguration(new_config)

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
