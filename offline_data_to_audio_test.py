import time
from multiprocessing import Queue, Process

import pandas as pd
from matplotlib import pyplot as plt

from custom_riffusion import SpectrogramConverter, SpectrogramParams
from data_utils import segment_eeg
from loops import transformer_w_img, player
from constants import FREQS


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

    eeg_queue, img_queue, audio_queue = Queue(), Queue(), Queue()

    transformer_process = Process(target=transformer_w_img, args=(eeg_queue, audio_queue, img_queue, FREQS, converter))
    player_process = Process(target=player, args=(audio_queue,))
    image_save_process = Process(target=img_saver, args=(img_queue,))

    transformer_process.start()
    player_process.start()
    image_save_process.start()

    data = pd.read_csv(data_path)
    segments = segment_eeg(data.to_numpy(),
                           sample_rate=250,
                           segment_len_s=5,
                           overlap_s=0)
    for segment in segments[:10]:
        eeg_queue.put(segment)
        time.sleep(0.1)
