import numpy as np
import skimage

from src.image_data.data_utils import leave_notes_in_spectrogram, get_sample_spectrogram, spectrogram_for_converter
from src.riffusion.custom_riffusion import SpectrogramImageConverter, SpectrogramParams
from src.audio_data.audio_utils import play_audio
from src.constants import NOTE2FREQ, NOTE_FREQUENCIES

from matplotlib import pyplot as plt


if __name__ == '__main__':

    brain_freq2note = {f: note for f, note in zip(NOTE_FREQUENCIES, NOTE2FREQ.keys())}
    MIN_NOTE_FREQ = NOTE2FREQ['C3 ']
    MAX_NOTE_FREQ = NOTE2FREQ['C7 ']

    print('Getting images...')
    sample_spectrogram = get_sample_spectrogram()
    skimage.io.imshow(sample_spectrogram)
    skimage.io.imsave('../../samples/sample_spectrogram.png', sample_spectrogram)

    notes = [NOTE2FREQ['C3 '], NOTE2FREQ['C4 '], NOTE2FREQ['E4 ']]
    sample_spectrogram_with_notes = leave_notes_in_spectrogram(sample_spectrogram, notes, MIN_NOTE_FREQ, MAX_NOTE_FREQ)
    skimage.io.imshow(sample_spectrogram_with_notes)
    skimage.io.imsave('../../samples/sample_spectrogram_with_cut_notes.png', sample_spectrogram_with_notes)

    print('Converting images to audio...')
    params = SpectrogramParams()
    params.min_frequency = MIN_NOTE_FREQ
    params.max_frequency = MAX_NOTE_FREQ
    converter = SpectrogramImageConverter(params=params)

    audio = converter.converter.audio_from_spectrogram(spectrogram_for_converter(sample_spectrogram))
    audio.export("../../samples/sample_audio.wav", format="wav")

    audio_cut_notes = converter.converter.audio_from_spectrogram(spectrogram_for_converter(sample_spectrogram_with_notes))
    audio_cut_notes.export("../../samples/sample_audio_cut_notes.wav", format="wav")

    # print('Playing...')
    # play_audio(audio)
