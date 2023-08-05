from src.data.eeg_features import extract_features
from src.data.spectral_transform import build_spectrogram_from_eeg_features
from src.data.utils import save_spectrogram_as_image
from src.data.torch_utils import SpectrogramConverter
from src.data.sample_gen import get_offline_eeg_segments

if __name__ == "__main__":
    converter = SpectrogramConverter()
    eeg_segments = get_offline_eeg_segments()

    for i, segment in enumerate(eeg_segments[:5]):
        features = extract_features(segment)
        spectrogram = build_spectrogram_from_eeg_features(features)
        save_spectrogram_as_image(spectrogram, f'../../samples/spectrograms/eeg_sample_{i}.png',
                                  inverse=False, flip=True)
