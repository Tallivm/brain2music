from pydub import AudioSegment

from src.data.utils import save_spectrogram_as_image
from src.data.torch_utils import SpectrogramConverter, SpectrogramParams


if __name__ == "__main__":
    sample_filepath = '../../samples/instruments/Bass-Drum-2.wav'
    test_audio = AudioSegment.from_wav(sample_filepath)
    params = SpectrogramParams()
    params.max_frequency = 10000
    converter = SpectrogramConverter(params=params)
    test_spectrogram = converter.spectrogram_from_audio(test_audio, use_mel=True)[0]
    save_spectrogram_as_image(test_spectrogram, '../../samples/instruments/Bass-Drum-2.png', inverse=True, flip=True)
