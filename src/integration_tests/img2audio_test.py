import numpy as np
import skimage

from src.riffusion.custom_riffusion import SpectrogramConverter, SpectrogramParams
from src.image_data.data_utils import normalize_img
from src.constants import NOTE2FREQ, RIFFUSION_MAX_POWER


if __name__ == "__main__":
    img = skimage.io.imread('../../samples/filters/threshold_niblack.png')
    img = normalize_img(img)
    img = np.expand_dims(img, 0).astype(np.float32)
    riff_spectrum = img * RIFFUSION_MAX_POWER

    min_freq = 0
    max_freq = NOTE2FREQ['C8 ']
    params = SpectrogramParams()
    params.min_frequency = min_freq
    params.max_frequency = max_freq
    converter = SpectrogramConverter(params=params)

    audio = converter.audio_from_spectrogram(riff_spectrum)
    audio.export("../../samples/filters/threshold_niblack_2.wav", format="wav")

