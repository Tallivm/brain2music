import numpy as np
import skimage


def resize_image(img: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    height = img.shape[0] if height is None else height
    width = img.shape[1] if width is None else width
    return skimage.transform.resize(img, (height, width), anti_aliasing=False)
