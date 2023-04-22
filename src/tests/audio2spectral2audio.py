from pydub import AudioSegment

from src.data.utils import save_pydub_audio_file
from src.data.torch_utils import SpectrogramConverter, SpectrogramParams

if __name__ == "__main__":

    params = SpectrogramParams()
    converter = SpectrogramConverter(params)
    test_audio = AudioSegment.from_wav('../../samples/sample_music.wav')
    test_spectrogram = converter.spectrogram_from_audio(test_audio, use_mel=False)[0]
    test_audio_back = converter.audio_from_spectrogram(test_spectrogram, use_mel=False)
    save_pydub_audio_file(test_audio_back, '../../samples/sample_music_back.wav')
