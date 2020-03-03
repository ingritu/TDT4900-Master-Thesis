from skimage.transform import resize
from skimage.io import imread
from skimage.io import imsave
from pathlib import Path
import numpy as np
import warnings

ROOT_PATH = Path(__file__).absolute().parents[2]


def resize_images(image_path, save_path, new_dims):
    """
    Resize images.

    Parameters
    ----------
    image_path : Path or str.
    save_path : Path or str.
    new_dims : list.
    """
    warnings.filterwarnings('ignore')
    image_path = Path(image_path)
    save_path = Path(save_path)
    directory = save_path.joinpath(str(new_dims[0]) + 'x' + str(new_dims[1]))
    if not directory.is_dir():
        directory.mkdir(parents=True)
    print("Resizing images")
    for im_file in image_path.glob('*.jpg'):
        p = len(str(im_file.parent)) + 1
        image_name = str(im_file)[p:]
        image = imread(str(im_file))
        image = resize(image, new_dims)
        image = image.astype(np.uint8)
        imsave(directory.joinpath(image_name), image)
