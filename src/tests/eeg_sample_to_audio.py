import time

from src.data.eeg_features import extract_features
from src.data.spectral_transform import build_spectrogram_from_eeg_features, transform_spectrogram, transform_wave
from src.data.utils import save_pydub_audio_file, save_spectrogram_as_image, produce_wave_with_torch, \
    produce_audio_from_wave
from src.data.torch_utils import SpectrogramConverter
from src.data.riffusion import load_stable_diffusion_img2img_pipeline
from src.data.sample_gen import get_sample_eeg_segment

if __name__ == "__main__":
    converter = SpectrogramConverter()
    sample_data = get_sample_eeg_segment()

    riffusion_model = load_stable_diffusion_img2img_pipeline()

    start = time.time()
    features = extract_features(sample_data)
    spectrogram = build_spectrogram_from_eeg_features(features)
    spectrogram = transform_spectrogram(spectrogram, riffusion_model=riffusion_model)

    wave = produce_wave_with_torch(spectrogram, converter)
    wave = transform_wave(wave)
    audio = produce_audio_from_wave(wave)

    end = time.time()

    print(f'Produced audio from a single sample of EEG data in {end - start:.2f} s')
    save_spectrogram_as_image(spectrogram, '../../samples/eeg_sample_to_audio.png')
    save_pydub_audio_file(audio, '../../samples/eeg_sample_to_audio.wav')

