import time
from multiprocessing import Queue
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.constants import SEGMENT_LEN_S, CHANNEL_IDS, SAMPLE_RATE
from src.data.eeg_features import clean_signal


FIG, AX = plt.subplots(figsize=(12, 6))

X = np.linspace(0, SEGMENT_LEN_S, SEGMENT_LEN_S * SAMPLE_RATE)
Y = np.zeros((SEGMENT_LEN_S * SAMPLE_RATE, len(CHANNEL_IDS)))

LNS = [
    AX.plot(X, np.empty(Y.shape[0]), lw=2)[0]
    for ch in range(len(CHANNEL_IDS))
]


def init_func():
    AX.set_ylim(-7.5, 25)
    AX.set_ylabel('')
    return LNS


def update(frame, img_queue):
    print('updating')
    global Y, LNS
    wave_arr = img_queue.get()
    wave_arr = wave_arr[:, CHANNEL_IDS]
    # for ch in range(wave_arr.shape[1]):
    #     wave_arr[:, ch] = clean_signal(wave_arr[:, ch])
    Y = np.vstack([Y[SAMPLE_RATE:, :], wave_arr])
    for ln, ch in zip(LNS, range(Y.shape[1])):
        data = Y[:, ch]
        data = (data - data.mean()) / data.std()
        ln.set_data(X, data + ch * 5)
    return LNS


def img_vizualizer(img_queue: Queue) -> None:
    time.sleep(10)
    animation = FuncAnimation(FIG, lambda x: update(x, img_queue),
                              interval=1000, init_func=init_func)
    plt.show()

