from src.data.utils import max_length_caption
from src.models.generator_framework import Generator

from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parents[2]


if __name__ == '__main__':
    # TODO: argparsing

    train_path = ROOT_PATH.joinpath('data',
                                    'interim',
                                    'karpathy_split',
                                    'flickr8k_mini_train.csv')
    val_path = ROOT_PATH.joinpath('data',
                                  'interim',
                                  'karpathy_split',
                                  'flickr8k_mini_val.csv')

    voc_path_ = ROOT_PATH.joinpath('data',
                                   'interim',
                                   'karpathy_split',
                                   'flickr8k_vocabulary.csv')
    feature_path_ = ROOT_PATH.joinpath('data',
                                       'processed',
                                       'Flickr8k',
                                       'Images',
                                       'encoded_visual_attention_full.pkl')
    # annotation file
    annFile = ROOT_PATH.joinpath('data',
                                 'processed',
                                 'annotations',
                                 'flickr8k_val.json')

    save_path_ = ROOT_PATH.joinpath('models')

    model_name_ = 'adaptive'

    batch_size = 2  # number of captions in a batch, not images
    # number of images in a batch, not captions during val
    val_batch_size = 1
    beam_size = 2

    epochs = 10
    em_dim = 50
    hidden_size_ = 50
    loss_function_ = 'cross_entropy'
    opt = 'adam'
    lr_ = 0.0001
    seed_ = 222

    # max_length = max_length_caption(train_path)
    max_length = 16

    input_shape_ = [[8, 8, 1536], max_length]

    generator = Generator(model_name_, input_shape_, hidden_size_,
                          voc_path_, feature_path_,
                          save_path_,
                          loss_function=loss_function_,
                          optimizer=opt, lr=lr_,
                          embedding_size=em_dim,
                          seed=seed_)
    generator.compile()

    # model is automatically saved after training
    generator.train(train_path,
                    val_path,
                    annFile,
                    epochs=epochs,
                    batch_size=batch_size,
                    beam_size=beam_size,
                    val_batch_size=val_batch_size)
