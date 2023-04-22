from multiprocessing import Queue, Process
from streamz import Stream

from src.data.eeg_features import extract_features
from src.data.spectral_transform import build_spectrogram_from_eeg_features, transform_spectrogram, transform_wave
from src.data.utils import produce_audio_from_wave, produce_wave_with_torch
from src.data.sample_gen import get_offline_eeg_segments
from src.data.torch_utils import SpectrogramConverter
from src.data.ai_models import load_rave_model
from src.player.player import player


def main_pipe(eeg_queue: Queue, play_queue: Queue) -> None:
    converter = SpectrogramConverter()
    rave_model = load_rave_model('darbouka_onnx')
    stream = Stream()
    (stream
     .map(extract_features)
     .map(build_spectrogram_from_eeg_features)
     .map(transform_spectrogram)
     .map(produce_wave_with_torch, converter)
     .map(transform_wave, rave_model, add_background_sound=True)
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

    main_pipe(eeg_queue, play_queue)
