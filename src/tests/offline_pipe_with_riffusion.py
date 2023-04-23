from torch.multiprocessing import Queue, Process

from src.pipe.pipe import main_pipe
from src.data.sample_gen import get_offline_eeg_segments
from src.player.player import player
from src.data.riffusion import load_stable_diffusion_img2img_pipeline


if __name__ == "__main__":
    eeg_queue, play_queue = Queue(), Queue()
    offline_segments = get_offline_eeg_segments()
    for segment in offline_segments:
        eeg_queue.put(segment)

    print(f'Will process {len(offline_segments)} pre-recorded EEG segments...')

    riffusion_model = load_stable_diffusion_img2img_pipeline()

    player_process = Process(target=player, args=(play_queue,))
    player_process.start()

    main_pipe(eeg_queue, play_queue, riffusion_model=riffusion_model)
