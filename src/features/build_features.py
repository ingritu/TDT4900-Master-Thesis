from pathlib import Path
from src.features.Resnet_features import extract_image_features
from src.features.resize_images import resize_images


ROOT_PATH = Path(__file__).absolute().parents[2]

# default image size for InceptionResnet is 299x299
DIMENSIONS = (299, 299, 3)

if __name__ == '__main__':
    # resize images
    datasets = ['Flickr8k']
    new_dims_ = (299, 299)
    for dataset_ in datasets:
        image_path_ = ROOT_PATH.joinpath('data', 'raw', dataset_,
                                         'Images')
        save_path_ = ROOT_PATH.joinpath('data', 'interim', dataset_,
                                        'Images')
        resize_images(image_path_, save_path_, new_dims_)

    # Build visual features
    datasets = ['Flickr8k']
    for dataset_ in datasets:
        image_path_ = ROOT_PATH.joinpath('data', 'interim', dataset_,
                                         'Images', str(DIMENSIONS[0]) + 'x'
                                         + str(DIMENSIONS[1]))
        save_path_ = ROOT_PATH.joinpath('data', 'processed', dataset_,
                                        'Images',
                                        'encoded_full_images.pkl')
        split_set_path_ = ROOT_PATH.joinpath('data', 'interim', dataset_,
                                             'Flickr8k_full.csv')

        extract_image_features(image_path_, save_path_, split_set_path_)
