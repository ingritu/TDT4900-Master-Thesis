from src.models.caption_generator import TutorialModel
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parents[2]


if __name__ == '__main__':
    train_path = ROOT_PATH.joinpath('data',
                                    'interim',
                                    'Flickr8k',
                                    'Flickr8k_train_clean.csv')
    val_path = ROOT_PATH.joinpath('data',
                                  'interim',
                                  'Flickr8k',
                                  'Flickr8k_val.csv')
    batch_size = 1
    epochs = 1




    model = TutorialModel()

    model.train(train_path, val_path, batch_size, epochs)
