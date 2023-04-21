import numpy as np
import skimage

from src.riffusion.custom_riffusion import SpectrogramConverter, SpectrogramParams
from src.image_data.data_utils import img_float2uint, spectrum2riff_spectrum, resize_image, normalize_img
from src.constants import NOTE2FREQ, VERTICAL_RESOLUTION, HORIZONTAL_RESOLUTION, SEGMENT_LEN_S, RIFFUSION_MAX_POWER


def assemble_spectrum_from_melody(melody: list[tuple[float, str]], width: int, height: int,
                                  time_s: float, freq_range: tuple[float, float],
                                  note2freq: dict[str, float]) -> np.ndarray:
    res = np.zeros((width//2, height//2))
    for time, note in melody:
        time_x = int(round(time / time_s * (width//2)))
        freq_y = int(round((note2freq[note] - freq_range[0]) / (freq_range[1] - freq_range[0]) * (height//2)))
        res[freq_y, time_x] = 1
    return resize_image(res, width, height)


if __name__ == '__main__':

    min_freq = 0
    max_freq = NOTE2FREQ['C8 ']
    params = SpectrogramParams()
    params.min_frequency = min_freq
    params.max_frequency = max_freq
    converter = SpectrogramConverter(params=params)

    melody = [
        # (time (s), note)
        (0.5, 'C4 '),
        (1, 'E4 '),
        (1.5, 'D4 '),
        (2, 'F4 '),
        (3, 'C4 '), (3, 'E4 '), (3, 'G4 '),
        (3.5, 'D4 '), (3.5, 'F4 '), (3.5, 'A4 '),
        (4, 'D4 '), (4, 'B4 '),
        (4.5, 'C4 '), (4.5, 'E4 '), (4.5, 'A#4'), (4.5, 'C5 ')
    ]

    fake_spectrum = assemble_spectrum_from_melody(melody, HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION, SEGMENT_LEN_S,
                                                  (min_freq, max_freq), NOTE2FREQ)
    fake_spectrum = spectrum2riff_spectrum(fake_spectrum)

    fake_img = img_float2uint(normalize_img(fake_spectrum))[0, :, :]
    skimage.io.imshow(fake_img)
    skimage.io.imsave('../../samples/filters/unfiltered.png', fake_img,
                      check_contrast=False)

    audio = converter.audio_from_spectrogram(fake_spectrum)
    audio.export("../../samples/filters/unfiltered.wav", format="wav")

    filters = {
        'gaussian3': (skimage.filters.gaussian, {'sigma': 3}),
        # 'gaussian10': (skimage.filters.gaussian, {'sigma': 10}),
        # 'hessian': (skimage.filters.hessian, {}),
        # 'meijering': (skimage.filters.meijering, {}),
        # 'sato': (skimage.filters.sato, {}),
        # 'scharr': (skimage.filters.scharr, {}),
        # 'threshold_niblack': (skimage.filters.threshold_niblack, {}),
        # 'frangi': (skimage.filters.frangi, {}),
    }

    for filter_name, (func, add_params) in filters.items():
        converter = SpectrogramConverter(params=params)
        filtered = func(fake_spectrum, **add_params)
        filtered = normalize_img(filtered)

        filtered_img = img_float2uint(filtered.copy())[0, :, :]
        skimage.io.imshow(filtered_img)
        skimage.io.imsave(f'../../samples/filters/{filter_name}.png', filtered_img,
                          check_contrast=False)

        audio = converter.audio_from_spectrogram(filtered * RIFFUSION_MAX_POWER)
        audio.export(f"../../samples/filters/{filter_name}.wav", format="wav")
