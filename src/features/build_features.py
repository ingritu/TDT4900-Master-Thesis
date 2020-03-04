from pathlib import Path
from src.features.resnet_features import extract_image_features
from src.features.resize_images import resize_images

import argparse


ROOT_PATH = Path(__file__).absolute().parents[2]

# default image size for InceptionResnet is 299x299
DIMENSIONS = (299, 299, 3)

if __name__ == '__main__':
    """
    To run script in terminal:
    python3 -m src.features.build_features
    """
    # resize images
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize-images', action='store_true',
                        help='Boolean to decide whether to resize the images '
                             'before building the actual features.')
    parser.add_argument('--new-image-size', type=int,
                        nargs='+',
                        help='List new image dimensions. should be something '
                             'like 299 299.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Which dataset to create image features for. '
                             'The options are '
                             '{flickr8k, flickr30k, coco}.')
    parser.add_argument('--karpathy', action='store_true',
                        help='Boolean used to decide whether to train on '
                             'the karpathy split of dataset or not.')
    parser.add_argument('--visual-attention', action='store_true',
                        help='Boolean for deciding whether to extract visual '
                             'features that are usable for models that use '
                             'visual attention.')
    parser.add_argument('--output-layer-idx', type=int, default=-3,
                        help='Which layer to extract features from.')
    parser.add_argument('--image-split', type=str, default='full',
                        help='Which dataset split to make features for. '
                             'Default value is full, meaning all images in '
                             'the dataset will be encoded and saved in the '
                             'same file.')
    args = vars(parser.parse_args())

    dataset_ = args['dataset']
    assert dataset_ in {'flickr8k', 'flickr30k', 'coco'}, \
        dataset_ + " is not supported. Only flickr8k, flickr30k and coco " \
                   "are supported."
    new_dims_ = args['new_image_size']
    dims = DIMENSIONS[:2]  # use default if nothing else is specified
    if len(new_dims_):
        assert len(new_dims_) in {2, 3}, \
            "new_image_size length out of range. " \
            "Expected length 2 or 3 but got " + str(len(new_dims_))
        # new dims are valid
        dims = new_dims_[:2]

    raw_path = ROOT_PATH.joinpath('data', 'raw')
    interim_path = ROOT_PATH.joinpath('data', 'interim')
    processed_path = ROOT_PATH.joinpath('data', 'processed')
    if args['resize_images']:
        # karpathy split does not matter here
        save_path_ = interim_path.joinpath(dataset_, 'Images')
        if dataset_ == 'flickr8k':
            image_path_ = raw_path.joinpath('Flickr8k', 'Images')
            resize_images(image_path_, save_path_, new_dims_)
        elif dataset_ == 'flickr30k':
            image_path_ = raw_path.joinpath('Flickr30k', dataset_ + '-images')
            resize_images(image_path_, save_path_, new_dims_)
        else:
            # must be coco
            # coco is special because there are three folders for the images
            image_dirs = ['train2014', 'test2014', 'val2014']
            for image_dir in image_dirs:
                image_path_ = raw_path.joinpath('MSCOCO', image_dir)
                resize_images(image_path_, save_path_, new_dims_)

    # Build visual features
    output_layer_dim_ = args['output_layer_idx']
    vis_att_ = args['visual_attention']
    split = args['image_split']
    image_path_ = interim_path.joinpath(dataset_,
                                        'Images', str(dims[0]) + 'x'
                                        + str(dims[1]))
    img_save_path = processed_path.joinpath('images')
    if args['karpathy']:
        img_save_path = img_save_path.joinpath('karpathy_split')
        interim_path = interim_path.joinpath('karpathy_split')
    file_str = dataset_ + '_encoded_'
    if vis_att_:
        file_str += 'visual_attention_'
    file_str += split + '.pkl'
    save_path_ = img_save_path.joinpath(file_str)
    split_set_path_ = interim_path.joinpath(dataset_ + '_' + split + '.csv')
    
    extract_image_features(image_path_,
                           save_path_,
                           split_set_path_,
                           output_layer_dim_,
                           vis_att=vis_att_)
