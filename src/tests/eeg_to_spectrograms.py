from src.data.eeg_features import extract_features
from src.data.spectral_transform import combine_spectrograms
from src.data.utils import save_spectrogram_as_image
from src.data.torch_utils import SpectrogramConverter
from src.data.sample_gen import get_offline_eeg_segments
from src.parameters import ChannelParameters
from src.constants import N_CHANNELS

if __name__ == "__main__":
    converter = SpectrogramConverter()
    eeg_segments = get_offline_eeg_segments()
    parameters = {i: ChannelParameters() for i in range(8)}

    for i, segment in enumerate(eeg_segments[:5]):
        spectrograms = []
        for ch in range(N_CHANNELS):
            spectrogram = extract_features(segment, ch=ch, channel_params=parameters[ch])
            spectrograms.append(spectrogram)
        spectrogram = combine_spectrograms(spectrograms)
        save_spectrogram_as_image(spectrogram, f'../../samples/spectrograms/eeg_sample_{i}.png',
                                  inverse=False, flip=True)
