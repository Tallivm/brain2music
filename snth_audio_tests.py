import numpy as np
from scipy import signal, ndimage
from scipy.io import wavfile
from matplotlib import pyplot as plt


if __name__ == "__main__":

    amplitude: int = np.iinfo(np.int16).max
    fs: int = 44100  # sampling rate
    duration: float = 2.0  # in seconds
    freq: float = 441.0  # sine frequency, Hz

    x = np.arange(fs * duration)

    waves = {
        'sine': amplitude * np.sin(2 * np.pi * freq * x / fs),
        'square': amplitude * signal.square(2 * np.pi * freq * x / fs, duty=0.5),
        'sawtooth': amplitude * signal.sawtooth(2 * np.pi * freq * x / fs, width=1),
        'fall_sawtooth': amplitude * signal.sawtooth(2 * np.pi * freq * x / fs, width=0),
        'triangle': amplitude * signal.sawtooth(2 * np.pi * freq * x / fs, width=0.5),
        'square_20': amplitude * signal.square(2 * np.pi * freq * x / fs, duty=0.2),
        'square_30': amplitude * signal.square(2 * np.pi * freq * x / fs, duty=0.3),
        'square_40': amplitude * signal.square(2 * np.pi * freq * x / fs, duty=0.4),
        'up_ladder': amplitude * signal.square(2 * np.pi * freq * x / fs, duty=0.5),
        'cycle_ladder': amplitude * signal.square(2 * np.pi * freq * x / fs, duty=0.5),
    }

    stairs = np.array(([1] * 25 + ([0] * 50 + [1] * 50) * (len(x) // 100))[:len(x)])
    small_stairs = np.array(([1] * 12 + ([0] * 25 + [1] * 25) * (len(x) // 50))[:len(x)])
    waves['up_ladder'] *= stairs
    waves['cycle_ladder'] *= small_stairs
    waves['complex_sine'] = waves['sine'] - ndimage.shift(waves['sine'] * -1, 25, mode='wrap')

    fig, ax = plt.subplots(len(waves), 1, figsize=(8, len(waves) * 2), sharex=True)

    for i, (name, wave) in enumerate(waves.items()):
        wavfile.write(f'samples/{name}.wav', fs, wave.astype(np.int16))

        ax[i].plot(wave)
        ax[i].set_title(name)

    plt.xlim(20000, 20500)
    plt.tight_layout()
    plt.savefig('samples/wave_shapes.png')
    plt.close()
