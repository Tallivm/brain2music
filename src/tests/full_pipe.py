from multiprocessing import Queue, Process

from src.pipe.pipe import main_pipe
from src.player.player import player
from src.data.riffusion import load_stable_diffusion_img2img_pipeline
from src.unicorn.unicorn import acquire_eeg
from src.constants import SAMPLE_RATE


if __name__ == "__main__":
    eeg_queue, play_queue = Queue(), Queue()

    riffusion_model = load_stable_diffusion_img2img_pipeline()

    recording_process = Process(target=acquire_eeg, args=(eeg_queue, SAMPLE_RATE))
    player_process = Process(target=player, args=(play_queue,))

    recording_process.start()
    player_process.start()

    main_pipe(eeg_queue, play_queue, riffusion_model=riffusion_model)



