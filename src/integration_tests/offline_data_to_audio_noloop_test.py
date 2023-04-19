import pandas as pd

from src.riffusion.custom_riffusion import SpectrogramConverter, SpectrogramParams
from src.image_data.data_utils import segment_eeg
from src.loops import eeg2img, img2audio, player
from src.constants import FREQUENCY

from queue import Queue


if __name__ == "__main__":
    data_path = '../../samples/UnicornRecorder_20220625_121622.csv'
    converter = SpectrogramConverter(SpectrogramParams())
    channels_to_acquire = [0, 1, 2]
    fs = 250

    data = pd.read_csv(data_path)
    segments = segment_eeg(data.to_numpy(),
                           sample_rate=250,
                           segment_len_s=5,
                           overlap_s=0)

    eeg_queue, img_queue, audio_queue = Queue(), Queue(), Queue()

    for segment in segments[:3]:
        eeg_queue.put(segment[:, channels_to_acquire])

    eeg2img(eeg_queue, img_queue, FREQUENCY, fs)
    img2audio(img_queue, audio_queue, converter)
    player(audio_queue)

