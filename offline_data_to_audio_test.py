import time
from multiprocessing import Queue, Process

import pandas as pd
from matplotlib import pyplot as plt

from custom_riffusion import SpectrogramConverter, SpectrogramParams
from data_utils import segment_eeg
from loops import eeg2img_loop, img2audio_loop, player
from constants import FREQUENCY


def img_saver(img_queue: Queue) -> None:
    i = 0
    while True:
        if not img_queue.empty():
            img = img_queue.get()
            print('Saving the image')
            plt.imshow(img)
            plt.savefig(f'samples/{i}_offline_spectrogram.png')
            plt.close()
            i += 1
        time.sleep(0.3)


if __name__ == "__main__":
    data_path = 'samples/UnicornRecorder_20220625_121622.csv'
    converter = SpectrogramConverter(SpectrogramParams())
    channels_to_acquire = [0, 1, 2]
    fs = 250

    data = pd.read_csv(data_path)
    segments = segment_eeg(data.to_numpy(),
                           sample_rate=250,
                           segment_len_s=5,
                           overlap_s=0)

    eeg_queue, img_queue, audio_queue = Queue(), Queue(), Queue()

    eeg2img_process = Process(target=eeg2img_loop, args=(eeg_queue, img_queue, FREQUENCY, fs))
    img2audio_process = Process(target=img2audio_loop, args=(img_queue, audio_queue, converter))
    player_process = Process(target=player, args=(audio_queue,))
    # image_save_process = Process(target=img_saver, args=(img_queue,))

    eeg2img_process.start()
    img2audio_process.start()
    player_process.start()
    # image_save_process.start()

    for segment in segments[:3]:
        eeg_queue.put(segment[:, channels_to_acquire])
        time.sleep(0.1)
