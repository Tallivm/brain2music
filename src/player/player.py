import time
from multiprocessing import Queue
from pydub import AudioSegment
from simpleaudio import play_buffer

from src.constants import PLAYBACK_SLEEP_TIME_S, SEGMENT_LEN_S


def play_audio_from_buffer(audio: AudioSegment, sleep_time: float = PLAYBACK_SLEEP_TIME_S) -> None:
    print(f'Playing {audio.duration_seconds} s audio (cut to {SEGMENT_LEN_S} s)...')
    play_buffer(audio[:SEGMENT_LEN_S * 1000].raw_data,
                num_channels=audio.channels,
                bytes_per_sample=audio.sample_width,
                sample_rate=audio.frame_rate
                )
    time.sleep(sleep_time)


def player(play_queue: Queue, sleep_time: float = PLAYBACK_SLEEP_TIME_S) -> None:
    while True:
        if not play_queue.empty():
            audio = play_queue.get()
            play_audio_from_buffer(audio, sleep_time=sleep_time)
        else:
            time.sleep(0.5)
