import numpy as np


class ChannelParameters:
    def __init__(self, min_freq: int = 4, max_freq: int = 50, volume: float = 1., abs_mode: str = 'both',
                 min_note_name: str = 'C2', max_note_name: str = 'C7', sample_rate: int = 250):
        self.min_freq: int = min_freq
        self.max_freq: int = max_freq
        self.min_note_name: str = min_note_name
        self.max_note_name: str = max_note_name
        self.volume: float = volume
        self.sample_rate: int = sample_rate
        self.abs_mode: str = abs_mode

        self.frequencies = np.arange(self.min_freq, self.max_freq + 1)
