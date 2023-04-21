from multiprocessing import Queue, Process

from src.loops import eeg2img_loop, img2audio_loop, player_loop
from src.unicorn.eeg_aquisition import acquire_eeg
from src.riffusion.custom_riffusion import SpectrogramConverter, SpectrogramParams
from src.constants import FREQUENCIES, SAMPLE_RATE


if __name__ == '__main__':

    # MAY BREAK ON GPU BECAUSE CUDA WORKS BADLY WITH MULTIPROCESSING!

    n_samples = SAMPLE_RATE
    converter = SpectrogramConverter(SpectrogramParams())
    channels_to_acquire = [0, 1, 2]

    eeg_queue, img_queue, audio_queue = Queue(), Queue(), Queue()

    acquisition_process = Process(target=acquire_eeg, args=(eeg_queue, n_samples, SAMPLE_RATE))
    eeg2img_process = Process(target=eeg2img_loop, args=(eeg_queue, img_queue, FREQUENCIES, SAMPLE_RATE))
    img2audio_process = Process(target=img2audio_loop, args=(img_queue, audio_queue, converter))
    player_process = Process(target=player_loop, args=(audio_queue,))

    acquisition_process.start()
    eeg2img_process.start()
    img2audio_process.start()
    player_process.start()
