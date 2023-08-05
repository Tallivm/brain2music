import time

from src.data.eeg_features import extract_features
from src.data.spectral_transform import combine_spectrograms, transform_spectrogram
from src.data.utils import save_pydub_audio_file, save_spectrogram_as_image, produce_audio_from_spectrogram_with_torch
from src.data.utils import apply_audio_filters
from src.data.torch_utils import SpectrogramConverter
from src.data.riffusion import load_stable_diffusion_img2img_pipeline
from src.data.sample_gen import get_sample_eeg_segment
from src.parameters import ChannelParameters
from src.constants import N_CHANNELS

if __name__ == "__main__":
    converter = SpectrogramConverter()
    sample_data = get_sample_eeg_segment()
    parameters = {i: ChannelParameters() for i in range(N_CHANNELS)}

    riffusion_model = load_stable_diffusion_img2img_pipeline()

    start = time.time()

    spectrograms = []
    for ch in range(N_CHANNELS):
        spectrogram = extract_features(sample_data, ch=ch, channel_params=parameters[ch])
        spectrograms.append(spectrogram)
    spectrogram = combine_spectrograms(spectrograms)
    save_spectrogram_as_image(spectrogram, '../../samples/eeg_sample_to_audio.png',
                              inverse=True, flip=True)

    spectrogram = transform_spectrogram(spectrogram, riffusion_model=riffusion_model, measure_difference=True)
    audio = produce_audio_from_spectrogram_with_torch(spectrogram, converter)
    audio = apply_audio_filters(audio)

    end = time.time()

    print(f'Produced audio from a single sample of EEG data in {end - start:.2f} s')
    # save_spectrogram_as_image(spectrogram, '../../samples/eeg_sample_to_audio_riff.png',
    #                           inverse=True, flip=True)
    # save_pydub_audio_file(audio, '../../samples/eeg_sample_to_audio_riff.wav')
