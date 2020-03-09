from pathlib import Path
from src.features.resnet_features import extract_image_features

import argparse


ROOT_PATH = Path(__file__).absolute().parents[2]

# default image size for InceptionResnet is 299x299
DIMENSIONS = (299, 299, 3)

if __name__ == '__main__':
    """
    To run script in terminal:
    python3 -m src.features.build_features
    """
    print("Started build features script.")
    parser = argparse.ArgumentParser()
    # build features
    parser.add_argument('--new-image-size', type=int,
                        nargs='+',
                        help='List new image dimensions. should be something '
                             'like 299 299.')
    parser.add_argument('--feature-split', type=str, default='full',
                        help='Which dataset split to make features for. '
                             'Default value is full, meaning all images in '
                             'the dataset will be encoded and saved in the '
                             'same file.')
    parser.add_argument('--karpathy', action='store_true',
                        help='Boolean used to decide whether to train on '
                             'the karpathy split of dataset or not.')
    parser.add_argument('--visual-attention', action='store_true',
                        help='Boolean for deciding whether to extract visual '
                             'features that are usable for models that use '
                             'visual attention.')
    parser.add_argument('--output-layer-idx', type=int, default=-3,
                        help='Which layer to extract features from.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Which dataset to create image features for. '
                             'The options are '
                             '{flickr8k, flickr30k, coco}.')
    args = vars(parser.parse_args())

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

    interim_path = ROOT_PATH.joinpath('data', 'interim')
    processed_path = ROOT_PATH.joinpath('data', 'processed')

    output_layer_dim_ = args['output_layer_idx']
    vis_att_ = args['visual_attention']
    split = args['feature_split']
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
    split_set_path_ = interim_path.joinpath(dataset_ + '_' + split +
                                            '.csv')
    print("Encoding images ...")
    extract_image_features(image_path_,
                           save_path_,
                           split_set_path_,
                           output_layer_dim_,
                           vis_att=vis_att_)
    print("Encoding done.")
