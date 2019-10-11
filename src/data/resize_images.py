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


if __name__ == '__main__':
    new_dims_ = (299, 299)
    image_path_ = ROOT_PATH.joinpath('data', 'raw', 'Flickr8k',
                                     'Images')
    save_path_ = ROOT_PATH.joinpath('data', 'interim', 'Flickr8k',
                                    'Images')
    resize_images(image_path_, save_path_, new_dims_)
