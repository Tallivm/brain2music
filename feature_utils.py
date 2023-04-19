import numpy as np


def get_sinewave(x, freq, fs, amplitude):
    return amplitude * np.sin(2 * np.pi * freq * x / fs)
