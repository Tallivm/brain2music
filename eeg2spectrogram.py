import numpy as np
import pydub.playback

import utils as u
from custom_riffusion import SpectrogramConverter, SpectrogramParams


# SAMPLE_DATA_PATH = '../../projects/brain_data/two_class_motor_imagery/S01T.mat'
# NEW_SAMPLE_RATE = 256
# WINDOW_SIZE_S = 5
# FREQS = np.arange(1, 32)
#
#
# test_data = u.load_eeg_mat_file(SAMPLE_DATA_PATH)[0]
# test_data_sample = u.resample_data(
#     eeg=test_data['X'],
#     old_sample_rate=test_data['fs'],
#     new_sample_rate=NEW_SAMPLE_RATE
# )
# test_segments = u.segment_eeg(
#     eeg=test_data_sample,
#     sample_rate=NEW_SAMPLE_RATE,
#     segment_len_s=WINDOW_SIZE_S,  # in seconds
#     overlap_s=0                   # in seconds
# )
# test_channel = test_segments[0][:, 0]
# test_spectrogram = u.wavelet_transform(
#     channel=test_channel,
#     freqs=FREQS,
#     sample_rate=NEW_SAMPLE_RATE
# )
# test_spectrogram = u.resize_image(
#     img=u.spectrogram_as_image(test_spectrogram, to_uint=False),
#     width=512,
#     height=512,
# )
#
#

print('Processing...')
converter = SpectrogramConverter(params=SpectrogramParams())

test_audio = pydub.AudioSegment.from_wav("../../other/Oriental_Dance.wav")
test_spectrograms = converter.spectrogram_from_audio(test_audio)

print('Received spectrograms...')

for i, spectrogram in enumerate(test_spectrograms):
    spectrogram = spectrogram.T
    spectrogram_parts = u.cut_array(spectrogram, len(spectrogram) - 512, 512)
    for j, part in enumerate(spectrogram_parts):
        print(part.shape)
        test_audio = converter.audio_from_spectrogram(part.T)
        print(f'Playing {i}/{j}...')
        pydub.playback.play(test_audio)
