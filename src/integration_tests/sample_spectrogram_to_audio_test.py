from multiprocessing import Queue
import skimage

from src.riffusion.custom_riffusion import SpectrogramConverter, SpectrogramParams
from src.image_data.data_utils import get_sample_spectrogram, spectrum2riff_spectrum
from src.loops import img2audio, player_loop


if __name__ == "__main__":
    converter = SpectrogramConverter(SpectrogramParams())
    sample_spectrogram = get_sample_spectrogram(abs_mode='both')
    skimage.io.imshow(sample_spectrogram)
    skimage.io.imsave('../../samples/sample_spectrogram.png', sample_spectrogram)

    riff_spectrum = spectrum2riff_spectrum(sample_spectrogram)
    skimage.io.imshow(riff_spectrum[0, :, :])
    skimage.io.imsave('../../samples/sample_riff_spectrogram.png', riff_spectrum[0, :, :])

    img_queue, audio_queue = Queue(), Queue()
    img_queue.put(riff_spectrum)
    img2audio(img_queue, audio_queue, converter)
    player_loop(audio_queue)

