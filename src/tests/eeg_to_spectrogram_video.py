import os
from pandas import read_csv
from tqdm import tqdm
import skimage

from src.data.eeg_features import extract_features
from src.data.spectral_transform import combine_spectrograms
from src.data.utils import save_spectrogram_as_image, segment_eeg, normalize_spectrogram_for_image
from src.data.torch_utils import SpectrogramConverter
from src.constants import SAMPLE_EEG_PATH, SAMPLE_RATE, SEGMENT_LEN_S, N_CHANNELS
from src.parameters import ChannelParameters


if __name__ == "__main__":
    converter = SpectrogramConverter()
    parameters = {i: ChannelParameters() for i in range(N_CHANNELS)}

    datapath = SAMPLE_EEG_PATH
    data = read_csv(datapath).to_numpy()[:, :N_CHANNELS]
    fps = 0.4
    eeg_segments = segment_eeg(data, sample_rate=SAMPLE_RATE, segment_len_s=SEGMENT_LEN_S,
                               overlap_s=SEGMENT_LEN_S - 1/fps)
    base_path = '../../../../readream/architecture'

    noises = {
        'gaussian': 'gaussian',
        'poisson': 'poisson',
        'snp': 's&p'
    }
    for color in ['black', 'white']:
        os.makedirs(f'{base_path}/{color}_clean/', exist_ok=True)
        for noise in noises:
            os.makedirs(f'{base_path}/{color}_{noise}/', exist_ok=True)

    for i, segment in tqdm(enumerate(eeg_segments), total=len(eeg_segments)):
        spectrograms = []
        for ch in range(N_CHANNELS):
            spectrogram = extract_features(segment, ch=ch, channel_params=parameters[ch])
            spectrograms.append(spectrogram)
        spectrogram = combine_spectrograms(spectrograms)

        for color in ['black', 'white']:
            save_spectrogram_as_image(spectrogram, f'{base_path}/{color}_clean/{i}.png',
                                      inverse=False if color == 'black' else True, flip=True, as_rgb=True)
            for noise_name, noise_str in noises.items():
                kwargs = {'amount': 0.01} if noise_name == 'snp' else {}
                noisy_spectrogram = skimage.util.random_noise(normalize_spectrogram_for_image(spectrogram),
                                                              mode=noise_str, seed=None, clip=True, **kwargs)
                save_spectrogram_as_image(noisy_spectrogram, f'{base_path}/{color}_{noise_name}/{i}.png',
                                          inverse=False if color == 'black' else True, flip=True, as_rgb=True)
