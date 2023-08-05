import os
import numpy as np
import skimage
import moviepy.video.io.ImageSequenceClip

from src.data.utils import natural_keys


if __name__ == '__main__':
    base_path = 'E:/Desktop/readream/architecture/out'

    color = 'black'
    suffix = 'clean'

    path_to_images = f'{base_path}/{color}_{suffix}/'
    imgs = [os.path.join(path_to_images, x) for x in os.listdir(path_to_images) if x.endswith('.png')]
    imgs = sorted(imgs, key=natural_keys)
    composite_img = skimage.io.imread(imgs[0])[:, :256]

    for img_path in imgs[1:]:
        img = skimage.io.imread(img_path)[:, :256]
        composite_img = np.hstack([composite_img, img])

    print('Saving...')
    skimage.io.imsave(f'{base_path}/composite_{color}_{suffix}.png', composite_img)
    print('Done')

    spectrogram_width = 512
    seconds_per_spectrogram = 5
    fps = 30

    shift = spectrogram_width / (seconds_per_spectrogram * fps)

    frames = []
    print('Collecting frames for video...')
    for i in range(0, composite_img.shape[1] - spectrogram_width, int(np.floor(shift))):
        frame = composite_img[:, i: i + spectrogram_width]
        frames.append(frame)
    print(f'Unique frame shapes: {set([x.shape for x in frames])}')
    print('Creating video...')
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=fps, with_mask=False)
    clip.write_videofile(f'{base_path}/composite_{color}_{suffix}.mp4')
    print('Done')
