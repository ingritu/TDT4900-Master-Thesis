from pathlib import Path
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from skimage.io import imread
import numpy as np
import pickle
from time import time
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]

# default image size for Resnet is 224x224
DIMENSIONS = (299, 299, 3)


def load_pre_trained_model():
    model = InceptionResNetV2(weights='imagenet')
    model_new = Model(model.input, model.layers[-2].output)
    model_new.summary()
    return model_new


def load_inception():
    model = InceptionV3(weights='imagenet')
    new_model = Model(model.input, model.layers[-2].output)
    new_model.summary()


def encode(image, model):
    # preprocess the image
    image = preprocess_input(image)
    # Get the encoding vector for the image
    fea_vec = model.predict(image)
    # reshape from (1, 2048) to (2048, )
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec


def extract_image_features(image_path, save_path, split_set_path):
    model = load_pre_trained_model()
    data_df = pd.read_csv(split_set_path)
    image_split = set(data_df.loc[:, 'image_id'])
    print(len(image_split))
    print(image_split)
    start = time()
    encoding_data = {}
    num_images = len(list(image_path.glob('*.jpg')))
    for im_file in image_path.glob('*.jpg'):
        p = len(str(im_file.parent)) + 1
        image_name = str(im_file)[p:]
        if image_name in image_split:
            encoding_data[im_file[num_images:]] = encode(imread(im_file),
                                                         model)
    print("Time taken in seconds =", time() - start)

    # Save the bottleneck train features to disk
    with open(save_path, "wb") as encoded_pickle:
        pickle.dump(encoding_data, encoded_pickle)


if __name__ == '__main__':
    dataset = 'Flickr8k'
    image_path_ = ROOT_PATH.joinpath('data', 'interim', dataset,
                                     'Images', str(DIMENSIONS[0]) + 'x'
                                     + str(DIMENSIONS[2]))
    save_path_ = ROOT_PATH.joinpath('data', 'processed', dataset,
                                    'Images',
                                    'encoded_val_images.pkl')
    split_set_path_ = ROOT_PATH.joinpath('data', 'interim', dataset,
                                         'Flickr8k_val.csv')

    extract_image_features(image_path_, save_path_, split_set_path_)
