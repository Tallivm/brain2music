import librosa as li
import skimage

from src.constants import SAMPLE_AUDIO_FILEPATH, NOTE2FREQ
from src.image_data.data_utils import img_float2uint, normalize_img, cut_array
from src.riffusion.custom_riffusion import SpectrogramConverter, SpectrogramParams


if __name__ == "__main__":
    wave, fs = li.load(SAMPLE_AUDIO_FILEPATH)
    spectrogram = li.feature.melspectrogram(y=wave, sr=fs, fmin=0, fmax=NOTE2FREQ['B8 '])
    print(f'Spectrogram in range: {spectrogram.min()} - {spectrogram.max()}')

    img = img_float2uint(normalize_img(spectrogram))
    skimage.io.imshow(img)
    skimage.io.imsave('../../samples/sample_audio_librosa_spectrogram.png', img)

    print(spectrogram.shape)
    converter = SpectrogramConverter(SpectrogramParams())
    segments = cut_array(spectrogram, round(fs * 5), 0)
    for i, segment in enumerate(segments):
        audio = converter.audio_from_spectrogram(segment)
        audio.export(f"../../samples/sample_audio_from_librosa_{i}.wav", format="wav")
