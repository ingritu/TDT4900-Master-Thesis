from pathlib import Path
from src.features.Resnet_features import extract_image_features
from src.features.resize_images import resize_images

import argparse


ROOT_PATH = Path(__file__).absolute().parents[2]

# default image size for InceptionResnet is 299x299
DIMENSIONS = (299, 299, 3)

if __name__ == '__main__':
    # resize images
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize_images', type=bool, default=False,
                        help='Boolean to decide whether to resize the images '
                             'before building the actual features.')
    parser.add_argument('--new_image_size', type=int, choices=range(300),
                        nargs='+',
                        help='List new image dimensions. should be something '
                             'like 299 299.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Which dataset to create image features for. '
                             'The options are '
                             '{flickr8k, flickr30k, coco}.')
    parser.add_argument('--visual_attention', type=bool, default=True,
                        help='Boolean for deciding whether to extract visual '
                             'features that are usable for models that use '
                             'visualt attention.')
    parser.add_argument('--output_layer_idx', type=int, default=-3,
                        help='Which layer to extract features from.')
    args = vars(parser.parse_args())

    dataset_ = args['dataset']
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
        image_path_ = raw_path.joinpath(dataset_, 'Images')
        save_path_ = interim_path.joinpath(dataset_, 'Images')
        resize_images(image_path_, save_path_, new_dims_)

    # Build visual features
    output_layer_dim_ = args['output_layer_idx']
    vis_att_ = args['visual_attention']
    image_path_ = interim_path.joinpath(dataset_,
                                        'Images', str(dims[0]) + 'x'
                                        + str(dims[1]))
    save_path_ = processed_path.joinpath(dataset_, 'Images',
                                         'encoded_visual_attention_full.pkl')
    split_set_path_ = interim_path.joinpath(dataset_, dataset_ + '_full.csv')

    extract_image_features(image_path_,
                           save_path_,
                           split_set_path_,
                           output_layer_dim_,
                           vis_att=vis_att_)
