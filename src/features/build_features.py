from pathlib import Path
from src.features.Resnet_features import extract_image_features


ROOT_PATH = Path(__file__).absolute().parents[2]

# default image size for Resnet is 224x224
DIMENSIONS = (299, 299, 3)

if __name__ == '__main__':
    # Build visual features
    dataset = 'Flickr8k'
    image_path_ = ROOT_PATH.joinpath('data', 'interim', dataset,
                                     'Images', str(DIMENSIONS[0]) + 'x'
                                     + str(DIMENSIONS[1]))
    save_path_ = ROOT_PATH.joinpath('data', 'processed', dataset,
                                    'Images',
                                    'encoded_full_images.pkl')
    split_set_path_ = ROOT_PATH.joinpath('data', 'interim', dataset,
                                         'Flickr8k_full.csv')

    extract_image_features(image_path_, save_path_, split_set_path_)
