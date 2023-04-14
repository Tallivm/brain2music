import logging
from multiprocessing import Process, Queue
import time

import numpy as np
from pydub import AudioSegment
import simpleaudio

import utils as u
from custom_riffusion import SpectrogramConverter, SpectrogramParams


CONVERTER = SpectrogramConverter(params=SpectrogramParams())


def fake_processing(spectrogram: np.ndarray, converter: SpectrogramConverter = CONVERTER) -> AudioSegment:
    audio = converter.audio_from_spectrogram(spectrogram.T)
    print('processed a spectrogram')
    return audio


def play_audio(audio: AudioSegment):
    print('playing a sound')
    simpleaudio.play_buffer(audio.raw_data,
                            num_channels=audio.channels,
                            bytes_per_sample=audio.sample_width,
                            sample_rate=audio.frame_rate
                            )
    time.sleep(5.09)


def transformer(eeg_buffer: Queue, audio_buffer: Queue):
    while True:
        if eeg_buffer:
            eeg_data = eeg_buffer.get()
            audio = fake_processing(eeg_data)
            audio_buffer.put(audio)
        time.sleep(0.01)


def player(audio_buffer: Queue):
    while True:
        if audio_buffer:
            audio = audio_buffer.get()
            play_audio(audio)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    logging.info('Loading sample WAV file...')
    test_audio = AudioSegment.from_wav("../../other/Oriental_Dance.wav")

    logging.info('Getting its spectrograms...')
    test_spectrogram = CONVERTER.spectrogram_from_audio(test_audio)[0]
    spectrogram_parts = u.cut_array(test_spectrogram.T, test_spectrogram.shape[1] - 512, 512)

    eeg_buffer, audio_buffer = Queue(), Queue()
    transformer_process = Process(target=transformer, args=(eeg_buffer, audio_buffer))
    player_process = Process(target=player, args=(audio_buffer,))

    player_process.start()
    transformer_process.start()

    for spectrogram in spectrogram_parts:
        eeg_buffer.put(spectrogram)
        time.sleep(0.1)
