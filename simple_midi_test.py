import numpy as np
from matplotlib import pyplot as plt
import skimage

from data_utils import wavelet_transform, resize_image, spectrogram_as_image
from feature_utils import get_sinewave
from custom_riffusion import SpectrogramConverter, SpectrogramParams
from audio_utils import play_audio, spectrogram2audio


if __name__ == '__main__':

    MIN_FREQ = 32
    MAX_FREQ = 1046
    BEAT_RESOLUTION = 8

    note2freq = {
        'C4 ': 261.63,
        'C#4': 277.18,
        'D4 ': 293.66,
        'D#4': 311.13,
        'E4 ': 329.63,
        'F4 ': 349.23,
        'F#4': 369.99,
        'G4 ': 392.00,
    }

    canvas = np.zeros((MAX_FREQ - MIN_FREQ, BEAT_RESOLUTION))

    canvas[round(note2freq['C4 '])] = np.array([1, 0, 0, 0, 1, 0, 1, 0])
    canvas[round(note2freq['D4 '])] = np.array([0, 0, 0, 0, 0, 1, 0, 0])
    canvas[round(note2freq['E4 '])] = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    canvas[round(note2freq['F4 '])] = np.array([0, 0, 1, 0, 0, 1, 0, 0])
    canvas[round(note2freq['G4 '])] = np.array([0, 0, 0, 1, 0, 0, 1, 0])

    # amplitude = np.iinfo(np.int16).max
    # fs = 44100
    # duration_s = 5
    #
    # x = np.linspace(0, fs * duration_s, 512)
    #
    # sinewaves = []
    # for _, freq in note2freq.items():
    #     sinewave = get_sinewave(x, freq, fs, amplitude)
    #     sinewaves.append(sinewave)
    #
    # sum_wave = np.mean(sinewaves, axis=0)
    # spectrogram = wavelet_transform(sum_wave,
    #                                 np.linspace(32, 1046, 512),
    #                                 fs)
    # spectrogram = resize_image(spectrogram, 512, 512)
    # spectrogram = spectrogram * resize_image(canvas, 512, 512)

    spectrogram = resize_image(canvas, 512, 512)
    spectrogram = spectrogram_as_image(spectrogram, to_uint=True)

    skimage.io.imshow(spectrogram, aspect='equal')
    skimage.io.imsave('samples/test_spectrogram.png', spectrogram)

    print('converting...')
    params = SpectrogramParams()
    params.min_frequency = 32
    params.max_frequency = 1046
    converter = SpectrogramConverter(params)
    audio = spectrogram2audio(spectrogram.copy(), converter)
    print('playing...')
    play_audio(audio)

