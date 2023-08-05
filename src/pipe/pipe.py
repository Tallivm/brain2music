import time
from torch.multiprocessing import Queue, Process

from streamz import Stream
from diffusers import StableDiffusionImg2ImgPipeline

from src.data.eeg_features import extract_all_features
from src.data.spectral_transform import transform_spectrogram, combine_spectrograms
from src.data.utils import apply_audio_filters, produce_audio_from_spectrogram_with_torch, save_audio_to_file
from src.data.sample_gen import get_offline_eeg_segments
from src.data.torch_utils import SpectrogramConverter
from src.player.player import player
from src.parameters import ChannelParameters
from src.constants import N_CHANNELS

from typing import Optional


def preprocessing_pipe(eeg_queue: Queue, spectra_queue: Queue, parameter_queue: Queue) -> None:
    stream = Stream()
    stream.map(extract_all_features).map(combine_spectrograms).sink(spectra_queue.put)

    while True:
        if not eeg_queue.empty():
            stream.emit((eeg_queue.get(), parameter_queue.get()))
        else:
            time.sleep(0.5)


def transform_spectrogram_pipe(spectra_queue: Queue, transformed_queue: Queue,
                               riffusion_model: Optional[StableDiffusionImg2ImgPipeline] = None,
                               measure_difference: bool = False) -> None:
    while True:
        if not spectra_queue.empty():
            spectrogram = spectra_queue.get()
            transformed = transform_spectrogram(spectrogram, riffusion_model, measure_difference)
            transformed_queue.put(transformed)
        else:
            time.sleep(0.5)


def afterparty_pipe(transformed_queue: Queue, play_queue: Queue) -> None:
    converter = SpectrogramConverter()
    stream = Stream()
    (stream
     .map(produce_audio_from_spectrogram_with_torch, converter)
     .map(apply_audio_filters)
     .accumulate(lambda acc, x: ((acc[0] + 1, x), (acc[0], x)), start=(0, 0), returns_state=True)
     .map(save_audio_to_file)
     .sink(play_queue.put))
    while True:
        if not transformed_queue.empty():
            stream.emit(transformed_queue.get())
        else:
            time.sleep(0.5)


def main_pipe(
    eeg_queue: Queue,
    play_queue: Queue,
    parameter_queue: Queue,
    riffusion_model: Optional[StableDiffusionImg2ImgPipeline] = None,
    measure_difference: bool = False
) -> None:
    spectra_queue, transformed_queue = Queue(), Queue()

    preprocessing_process = Process(target=preprocessing_pipe, args=(eeg_queue, spectra_queue, parameter_queue))
    afterparty_process = Process(target=afterparty_pipe, args=(transformed_queue, play_queue))

    preprocessing_process.start()
    afterparty_process.start()

    transform_spectrogram_pipe(spectra_queue, transformed_queue, riffusion_model, measure_difference=measure_difference)


if __name__ == "__main__":
    eeg_queue, play_queue, parameter_queue = Queue(), Queue(), Queue()
    offline_segments = get_offline_eeg_segments()
    for segment in offline_segments:
        eeg_queue.put(segment)
        parameter_queue.put({i: ChannelParameters() for i in range(N_CHANNELS)})

    print(f'Will process {eeg_queue.qsize()} segments')

    player_process = Process(target=player, args=(play_queue,))
    player_process.start()

    main_pipe(eeg_queue, play_queue, parameter_queue)

