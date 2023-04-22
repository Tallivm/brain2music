import time

from src.data.eeg_features import extract_features
from src.data.spectral_transform import build_spectrogram_from_eeg_features, transform_spectrogram
from src.data.utils import produce_audio_from_spectrogram_with_librosa, get_sample_eeg_segment, save_pydub_audio_file, \
    save_spectrogram_as_image, normalize_spectrogram_with_max_power


if __name__ == "__main__":
    sample_data = get_sample_eeg_segment()

    start = time.time()
    features = extract_features(sample_data)
    spectrogram = build_spectrogram_from_eeg_features(features)
    spectrogram = transform_spectrogram(spectrogram)
    audio = produce_audio_from_spectrogram_with_librosa(normalize_spectrogram_with_max_power(spectrogram))
    end = time.time()

    print(f'Produced audio from a single sample of EEG data in {end - start:.2f} s')
    save_spectrogram_as_image(spectrogram, '../../samples/eeg_sample_to_audio.png')
    save_pydub_audio_file(audio, '../../samples/eeg_sample_to_audio.wav')

    for ch in range(len(features)):
        save_spectrogram_as_image(features[ch], f'../../samples/eeg_sample_to_audio_ch{ch}.png')
