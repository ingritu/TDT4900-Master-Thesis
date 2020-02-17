from pathlib import Path
from src.data.utils import max_length_caption
from src.models.generator_framework import Generator

ROOT_PATH = Path(__file__).absolute().parents[2]

if __name__ == '__main__':
    # TODO: argparsing
    train_path = ROOT_PATH.joinpath('data',
                                    'interim',
                                    'Flickr8k',
                                    'Flickr8k_train_clean.csv')
    val_path = ROOT_PATH.joinpath('data',
                                  'interim',
                                  'Flickr8k',
                                  'Flickr8k_mini_test.csv')

    voc_path_ = ROOT_PATH.joinpath('data',
                                   'interim',
                                   'Flickr8k',
                                   'Flickr8k_vocabulary.csv')
    feature_path_ = ROOT_PATH.joinpath('data',
                                       'processed',
                                       'Flickr8k',
                                       'Images',
                                       'encoded_visual_attention_full.pkl')
    saved_model_path_ = ROOT_PATH.joinpath(
        'models',
        'adaptive_17-Feb-2020_(15:50:13)',
        'BEST_checkpoint_1.pth.tar')

    save_path_ = ROOT_PATH.joinpath('models')

    model_name_ = 'adaptive'

    em_dim = 300
    loss_function_ = 'cross_entropy'
    opt = 'adam'
    lr_ = 0.0001
    seed_ = 222

    max_length = max_length_caption(train_path)

    input_shape_ = [[8, 8, 1536], max_length]
    hidden_shape_ = 50

    generator = Generator(model_name_, input_shape_, hidden_shape_,
                          voc_path_, feature_path_,
                          save_path_,
                          loss_function=loss_function_,
                          optimizer=opt, lr=lr_,
                          embedding_size=em_dim,
                          seed=seed_)

    generator.load_model(saved_model_path_)

    print('Making predictions!')
    beam_size_ = 3
    print('Using beam Size =', beam_size_)
    result = generator.predict(val_path, beam_size=beam_size_)

    print(result)




