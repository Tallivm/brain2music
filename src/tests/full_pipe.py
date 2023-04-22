from multiprocessing import Queue, Process

from src.pipe.pipe import main_pipe
from src.player.player import player
from src.data.riffusion import load_stable_diffusion_img2img_pipeline, get_generator
from src.unicorn.unicorn import acquire_eeg
from src.data.ai_models import load_rave_model
from src.constants import SAMPLE_RATE


if __name__ == "__main__":
    eeg_queue, play_queue = Queue(), Queue()

    # riffusion_model = load_stable_diffusion_img2img_pipeline()
    # generator = get_generator(42, 'cuda')
    rave_model = load_rave_model('vintage')

    recording_process = Process(target=acquire_eeg, args=(eeg_queue, SAMPLE_RATE))
    player_process = Process(target=player, args=(play_queue,))

    recording_process.start()
    player_process.start()

    # main_pipe(eeg_queue, play_queue, riffusion_model=riffusion_model, generator=generator)
    main_pipe(eeg_queue, play_queue, rave_model=rave_model)
