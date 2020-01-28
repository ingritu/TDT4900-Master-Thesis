from pathlib import Path
from src.models.caption_generator import TutorialModel
from src.data.max_length_caption import max_length_caption

ROOT_PATH = Path(__file__).absolute().parents[2]

if __name__ == '__main__':
    train_path = ROOT_PATH.joinpath('data',
                                    'interim',
                                    'Flickr8k',
                                    'Flickr8k_train_clean.csv')
    val_path = ROOT_PATH.joinpath('data',
                                  'interim',
                                  'Flickr8k',
                                  'Flickr8k_mini_test.csv')


    model_path = ROOT_PATH.joinpath(
        'models',
        'Tutorial_28-Jan-2020 (14:03:29.752294)_.h5')

    batch_size = 300
    epochs = 1
    em_dim = 300
    pre_trained = False
    max_length = max_length_caption(train_path)

    model = TutorialModel(max_length, embedding_dim=em_dim,
                          pre_trained_embeddings=pre_trained)

    model.compile(weights=model_path)

    result = model.predict_greedy(val_path)
    print(result)



