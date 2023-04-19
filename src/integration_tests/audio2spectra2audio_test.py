import logging
from multiprocessing import Process, Queue
import time

import numpy as np
from pydub import AudioSegment

from src.image_data import data_utils as u
from src.riffusion.custom_riffusion import SpectrogramConverter, SpectrogramParams
from src.loops import player


def fake_processing(spectrogram: np.ndarray, converter: SpectrogramConverter) -> AudioSegment:
    audio = converter.audio_from_spectrogram(spectrogram.T)
    print('processed a spectrogram')
    return audio


def fake_transformer(eeg_queue: Queue, audio_queue: Queue, converter: SpectrogramConverter) -> None:
    while True:
        if eeg_queue:
            eeg_data = eeg_queue.get()
            assert eeg_data.shape == (512, 512)
            audio = fake_processing(eeg_data, converter)
            audio_queue.put(audio)
        time.sleep(0.01)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    converter = SpectrogramConverter(params=SpectrogramParams())

    logging.info('Loading sample WAV file...')
    test_audio = AudioSegment.from_wav("../../samples/Oriental_Dance.wav")

    logging.info('Getting its spectrograms...')
    test_spectrogram = converter.spectrogram_from_audio(test_audio)[0]
    spectrogram_parts = u.cut_array(test_spectrogram.T, test_spectrogram.shape[1] - 512, 512)

    eeg_queue, audio_queue = Queue(), Queue()
    transformer_process = Process(target=fake_transformer, args=(eeg_queue, audio_queue, converter))
    player_process = Process(target=player, args=(audio_queue,))

    player_process.start()
    transformer_process.start()

    for spectrogram in spectrogram_parts:
        eeg_queue.put(spectrogram)
        time.sleep(0.1)
