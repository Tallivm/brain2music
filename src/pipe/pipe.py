from multiprocessing import Queue, Process
from streamz import Stream
from torch import Generator
from diffusers import StableDiffusionImg2ImgPipeline

from src.data.eeg_features import extract_features
from src.data.spectral_transform import build_spectrogram_from_eeg_features, transform_spectrogram, transform_wave
from src.data.utils import produce_audio_from_wave, produce_wave_with_torch
from src.data.sample_gen import get_offline_eeg_segments
from src.data.torch_utils import SpectrogramConverter
from src.data.ai_models import load_rave_model
from src.player.player import player

from typing import Optional


def main_pipe(
    eeg_queue: Queue,
    play_queue: Queue,
    riffusion_model: Optional[StableDiffusionImg2ImgPipeline] = None,
    generator: Optional[Generator] = None,
    rave_model: Optional = None,
    add_background_sound: bool = False
) -> None:
    converter = SpectrogramConverter()
    stream = Stream()
    (stream
     .map(extract_features)
     .map(build_spectrogram_from_eeg_features)
     .map(transform_spectrogram, riffusion_model=riffusion_model, generator=generator)
     .map(produce_wave_with_torch, converter)
     .map(transform_wave, rave_model=rave_model, add_background_sound=add_background_sound)
     .map(produce_audio_from_wave)
     .sink(play_queue.put)
     )

    while True:
        stream.emit(eeg_queue.get())


if __name__ == "__main__":
    eeg_queue, play_queue = Queue(), Queue()
    offline_segments = get_offline_eeg_segments()
    for segment in offline_segments:
        eeg_queue.put(segment)

    player_process = Process(target=player, args=(play_queue,))
    player_process.start()

    rave_model = load_rave_model('darbouka_onnx')
    main_pipe(eeg_queue, play_queue, rave_model=rave_model)
