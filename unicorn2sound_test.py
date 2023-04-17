from multiprocessing import Queue, Process

from loops import transformer, player, acquire_eeg
from custom_riffusion import SpectrogramConverter, SpectrogramParams


if __name__ == '__main__':

    RECORD_SIZE_S = 1
    converter = SpectrogramConverter(SpectrogramParams())
    channels_to_acquire = [0, 1, 2]

    eeg_queue, audio_queue = Queue(), Queue()

    acquisition_process = Process(target=acquire_eeg, args=(eeg_queue, RECORD_SIZE_S, channels_to_acquire))
    transformer_process = Process(target=transformer, args=(eeg_queue, audio_queue, converter))
    player_process = Process(target=player, args=(audio_queue,))

    acquisition_process.start()
    transformer_process.start()
    player_process.start()
