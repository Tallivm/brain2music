from multiprocessing import Queue
import skimage

from src.riffusion.custom_riffusion import SpectrogramConverter, SpectrogramParams
from src.image_data.data_utils import spectrum2riff_spectrum, normalize_img
from src.loops import img2audio, player_loop
from src.constants import NOTE2FREQ


if __name__ == "__main__":
    img = skimage.io.imread('../../samples/filters/threshold_niblack.png')
    img = normalize_img(img)

    min_freq = 0
    max_freq = NOTE2FREQ['C8 ']
    params = SpectrogramParams()
    params.min_frequency = min_freq
    params.max_frequency = max_freq
    converter = SpectrogramConverter(params=params)

    riff_spectrum = spectrum2riff_spectrum(img)

    img_queue, audio_queue = Queue(), Queue()
    img_queue.put(riff_spectrum)
    img2audio(img_queue, audio_queue, converter)
    # player_loop(audio_queue)
    audio = audio_queue.get()
    audio.export("../../samples/filters/threshold_niblack_2.wav", format="wav")

