import pydub.playback
import utils as u
from custom_riffusion import SpectrogramConverter, SpectrogramParams
import logging
from tqdm import tqdm


if __name__ == '__main__':

    logging.info('Initializing Spectrogram Converter...')
    converter = SpectrogramConverter(params=SpectrogramParams())

    logging.info('Loading sample WAV file...')
    test_audio = pydub.AudioSegment.from_wav("../../other/Oriental_Dance.wav")

    logging.info('Getting its spectrogram...')
    test_spectrograms = converter.spectrogram_from_audio(test_audio)

    for i, spectrogram in enumerate(test_spectrograms):
        logging.info(f'Working with spectrogram {i}...')
        spectrogram_parts = u.cut_array(spectrogram.T, spectrogram.shape[1] - 512, 512)
        audios = []
        for part in tqdm(spectrogram_parts, desc='Processing parts'):
            test_audio = converter.audio_from_spectrogram(part.T)
            audios.append(test_audio)

        logging.info(f'Playing spectrogram {i}...')
        for audio in audios:
            pydub.playback.play(audio)
