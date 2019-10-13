from skimage.transform import resize
from skimage.io import imread
from skimage.io import imsave
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parents[2]


def resize_images(image_path, save_path, new_dims):
    for im_file in image_path.glob('*.jpg'):
        p = len(str(im_file.parent)) + 1
        image_name = str(im_file)[p:]
        image = imread(im_file)
        image = resize(image, new_dims)
        imsave(save_path.joinpath(str(new_dims[0]) + 'x' +
                                  str(new_dims[1]),
                                  image_name), image)
