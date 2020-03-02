from pathlib import Path
from skimage.io import imread
import numpy as np
from pickle import dump
from pickle import load
from time import time
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]


def load_pre_trained_model(output_layer_idx):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.models import Model
    model = InceptionResNetV2(weights='imagenet')
    model_new = Model(model.input, model.layers[output_layer_idx].output)
    model_new.summary()
    return model_new


def load_inception():
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model
    model = InceptionV3(weights='imagenet')
    new_model = Model(model.input, model.layers[-2].output)
    new_model.summary()
    return new_model


def encode(image, model):
    from keras.applications.inception_resnet_v2 import preprocess_input
    # add one more dimension
    image = np.expand_dims(image, axis=0)
    # preprocess the image
    image = preprocess_input(image)
    # Get the encoding vector for the image
    fea_vec = model.predict(image)
    # reshape from (1, 2048) to (2048, )
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec


def encode_vis_att(image, model):
    from keras.applications.inception_resnet_v2 import preprocess_input
    # add one more dimension
    image = np.expand_dims(image, axis=0)
    # preprocess the image
    image = preprocess_input(image)
    # Get the encoding vector for the image
    fea_vec = model.predict(image)
    # reshape from (1, 8, 8, 1536) to (8, 8, 1536)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1:])
    return fea_vec


def extract_image_features(image_path,
                           save_path,
                           split_set_path,
                           output_layer_idx,
                           vis_att=True):
    # consider splitting the process up in parts and then
    # combining the parts at the end, to reduce the amount of images
    # in memory at any time
    image_path = Path(image_path)
    save_path = Path(save_path)
    save_dir = save_path.parent
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    model = load_pre_trained_model(output_layer_idx)
    data_df = pd.read_csv(split_set_path)
    image_split = set(data_df.loc[:, 'image_id'])
    start = time()
    encoding_data = {}
    n = len(image_split)
    count = 0
    for im_file in image_path.glob('*.jpg'):
        p = len(str(im_file.parent)) + 1
        im_file = str(im_file)
        image_name = im_file[p:]
        if image_name in image_split:
            count += 1
            if vis_att:
                encoding_data[image_name] = encode_vis_att(imread(im_file),
                                                           model)
            else:
                encoding_data[image_name] = encode(imread(im_file), model)
            print(str(count) + ' / ' + str(n))
    print("Time taken in seconds =", time() - start)
    # Save the bottleneck train features to disk
    with open(save_path, "wb") as encoded_pickle:
        dump(encoding_data, encoded_pickle)


def load_visual_features(feature_path):
    with open(feature_path, 'rb') as file:
        data_features = load(file)
    print('Photos: %d' % len(data_features))
    return data_features
