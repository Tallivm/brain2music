from torch.multiprocessing import Queue, Process

from src.pipe.pipe import main_pipe
from src.data.sample_gen import get_offline_eeg_segments
from src.player.player import player
from src.parameters import ChannelParameters
from src.constants import N_CHANNELS
from src.tests.dash_app import start_dash


def main_pipe_start(eeg_queue, play_queue, parameter_queue):
    main_pipe(eeg_queue, play_queue, parameter_queue, riffusion_model=None, measure_difference=True)


if __name__ == "__main__":
    eeg_queue, img_queue, play_queue, parameter_queue = Queue(), Queue(), Queue(), Queue()

    offline_segments = get_offline_eeg_segments()
    for segment in offline_segments:
        eeg_queue.put(segment)
        for i in range(5):
            img_queue.put(segment[i * 250:(i + 1) * 250])
            parameter_queue.put({i: ChannelParameters() for i in range(N_CHANNELS)})

    print(f'Will process {len(offline_segments)} pre-recorded EEG segments...')

    player_process = Process(target=player, args=(play_queue,))
    main_pipe_process = Process(target=main_pipe_start, args=(eeg_queue, play_queue, parameter_queue))

    player_process.start()
    main_pipe_process.start()

    start_dash(parameter_queue)
