from src.models.caption_generator import model_switcher
from src.data.max_length_caption import max_length_caption
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parents[2]


if __name__ == '__main__':
    model_str_ = 'TestModel'
    train_path = ROOT_PATH.joinpath('data',
                                    'interim',
                                    'Flickr8k',
                                    'Flickr8k_train_clean.csv')
    val_path = ROOT_PATH.joinpath('data',
                                  'interim',
                                  'Flickr8k',
                                  'Flickr8k_val.csv')
    batch_size = 300
    epochs = 1
    em_dim = 300
    pre_trained = False
    max_length = max_length_caption(train_path)

    model = model_switcher(model_str_, max_length)
    model.compile()

    model.train(train_path, val_path, batch_size, epochs)
