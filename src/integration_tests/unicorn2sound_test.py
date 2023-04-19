from multiprocessing import Queue, Process

from src.loops import eeg2img_loop, img2audio_loop, player, acquire_eeg
from src.riffusion.custom_riffusion import SpectrogramConverter, SpectrogramParams
from src.constants import FREQUENCY


if __name__ == '__main__':

    RECORD_SIZE_S = 1
    converter = SpectrogramConverter(SpectrogramParams())
    channels_to_acquire = [0, 1, 2]
    fs = 250

    eeg_queue, img_queue, audio_queue = Queue(), Queue(), Queue()

    acquisition_process = Process(target=acquire_eeg, args=(eeg_queue, RECORD_SIZE_S, fs))
    eeg2img_process = Process(target=eeg2img_loop, args=(eeg_queue, img_queue, FREQUENCY, fs))
    img2audio_process = Process(target=img2audio_loop, args=(img_queue, audio_queue, converter))
    player_process = Process(target=player, args=(audio_queue,))

    acquisition_process.start()
    eeg2img_process.start()
    img2audio_process.start()
    player_process.start()
