from skimage.transform import resize
from skimage.io import imread
from skimage.io import imsave
from pathlib import Path
import numpy as np
import argparse

ROOT_PATH = Path(__file__).absolute().parents[2]

# default image size for InceptionResnet is 299x299
DIMENSIONS = (299, 299, 3)


def resize_images(image_path, save_path, new_dims):
    '''
    Resize images.

    Parameters
    ----------
    image_path : Path or str.
    save_path : Path or str.
    new_dims : list.
    '''
    image_path = Path(image_path)
    save_path = Path(save_path)
    directory = save_path.joinpath(str(new_dims[0]) + 'x' + str(new_dims[1]))
    if not directory.is_dir():
        directory.mkdir(parents=True)
    print("Resizing images")
    count = 0
    for im_file in image_path.glob('*.jpg'):
        p = len(str(im_file.parent)) + 1
        image_name = str(im_file)[p:]
        image = imread(str(im_file))
        image = resize(image, new_dims)
        image = image*255
        imsave(str(directory.joinpath(image_name)), image.astype(np.uint8))
        count += 1
        if count % 1000 == 0:
            print(count)


if __name__ == '__main__':
    """
    To run script in terminal:
    python3 -m src.features.resize_images
    """
    print("Started resize images script.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize-images', action='store_true',
                        help='Boolean to decide whether to resize the images '
                             'before building the actual features.')
    parser.add_argument('--new-image-size', type=int,
                        nargs='+',
                        help='List new image dimensions. should be something '
                             'like 299 299.')
    parser.add_argument('--image-split', type=str, default='full',
                        help='Which dataset split images to resize. '
                             'Default is full, meaning all images in '
                             'the dataset will be resized. This is only '
                             'necessary for coco since it is so big.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Which dataset to create image features for. '
                             'The options are '
                             '{flickr8k, flickr30k, coco}.')

    args = vars(parser.parse_args())
    print("Past argparse.")
    # print all args
    print("using parsed arguments.")
    for key in args:
        print(key, args[key])


    dataset_ = args['dataset']
    assert dataset_ in {'flickr8k', 'flickr30k', 'coco'}, \
        dataset_ + " is not supported. Only flickr8k, flickr30k and coco " \
                   "are supported."
    new_dims_ = args['new_image_size']
    dims = DIMENSIONS  # use default if nothing else is specified
    if len(new_dims_):
        assert len(new_dims_) in {2, 3}, \
            "new_image_size length out of range. " \
            "Expected length 2 or 3 but got " + str(len(new_dims_))
        # new dims are valid
        if len(new_dims_) == 2:
            dims = new_dims_[:2] + [3]
        else:
            dims = new_dims_
    """""""""
    raw_path = ROOT_PATH.joinpath('data', 'raw')
    interim_path = ROOT_PATH.joinpath('data', 'interim')
    processed_path = ROOT_PATH.joinpath('data', 'processed')
    if args['resize_images']:
        # karpathy split does not matter here
        save_path_ = interim_path.joinpath(dataset_, 'Images')
        if dataset_ == 'flickr8k':
            image_path_ = raw_path.joinpath('Flickr8k', 'Images')
            resize_images(image_path_, save_path_, dims)
        elif dataset_ == 'flickr30k':
            image_path_ = raw_path.joinpath('Flickr30k', dataset_ + '-images')
            resize_images(image_path_, save_path_, dims)
        else:
            # must be coco
            # coco is special because there are three folders for the images
            image_dirs = ['train2014', 'test2014', 'val2014']
            img_split = args['image_split']
            assert img_split in {'train', 'test', 'val'}, \
                "Illegal image split. Must be either train, test or val."
            image_path_ = raw_path.joinpath('MSCOCO', img_split + '2014')
            resize_images(image_path_, save_path_, dims)
            print("Resizing images done.")
    """""""""
    print("End of Script.")
