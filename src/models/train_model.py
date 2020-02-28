from src.data.utils import max_length_caption
from src.models.generator_framework import Generator

from pathlib import Path
import argparse

ROOT_PATH = Path(__file__).absolute().parents[2]


if __name__ == '__main__':
    """
    To run script in terminal:
    python3 -m src.models.train_model --args
    """
    # All default values are the values used in the knowing when to look paper
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size. '
                             'The number of captions in a batch.')
    parser.add_argument('--val_batch_size', type=int, default=250,
                        help='Validation batch size. '
                             'The number of images in a batch. '
                             'The actual batch size is val_batch_size * '
                             'beam_size.')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='Beam size to use in beam search '
                             'inference algorithm. '
                             'Bigger beam size yields higher performance.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='The number of epochs to train the network for.')
    parser.add_argument('--embedding_size', type=int, default=512,
                        help='Embedding dimension. '
                             'The size of the word vector representations.')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden dimension.')
    parser.add_argument('--loss_function', type=str, default='cross_entropy',
                        help='Loss/Cost function to use during training.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use during training.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate for the decoder.')
    parser.add_argument('--seed', type=int, default=222,
                        help='Random state seed.')
    parser.add_argument('--model', type=str, default='adaptive_decoder',
                        help='Model name. Which model type to train.')
    parser.add_argument('--karpathy', type=bool, default=True,
                        help='Boolean used to decide whether to train on '
                             'the karpathy split of dataset or not.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset to train on. The options are '
                             '{flickr8k, flickr30k, coco}.')
    parser.add_argument('--image_feature_size', type=int,
                        nargs='+', required=True,
                        help='List integers. Should be something like '
                             '--image_feature_size 8 8 1536.')
    # there still are more customizable parameters to set,
    # add these later
    args = vars(parser.parse_args())  # access args as dictionary

    interim_path = ROOT_PATH.joinpath('data',
                                      'interim')
    processed_path = ROOT_PATH.joinpath('data',
                                        'processed')
    dataset = args['dataset']
    if args['karpathy']:
        interim_path = interim_path.joinpath('karpathy_split')
        # annotation file
        annFile = processed_path.joinpath('annotations',
                                          'karpathy_split',
                                          dataset + '_val.json')

    train_path = interim_path.joinpath(dataset + '_train_clean.csv')
    val_path = interim_path.joinpath(dataset + '_val.csv')
    voc_path_ = interim_path.joinpath(dataset + '_vocabulary.csv')
    feature_path_ = processed_path.joinpath(
        dataset, 'Images', 'encoded_visual_attention_full.pkl')

    save_path_ = ROOT_PATH.joinpath('models')

    model_name_ = args['model']

    batch_size = args['batch_size']
    val_batch_size = args['val_batch_size']
    beam_size = args['beam_size']

    epochs = args['epochs']
    em_dim = args['embedding_size']
    hidden_size_ = args['hidden_size']
    loss_function_ = args['loss_function']
    opt = args['optimizer']
    lr_ = args['lr']
    seed_ = args['seed']

    max_length = max_length_caption(train_path)

    image_feature_size = args['image_feature_size']
    assert len(image_feature_size) == 3, "Wrong argument length for " \
                                         "image_feature_size. " \
                                         "Expected 3 but got " + \
                                         str(len(image_feature_size)) + "."
    input_shape_ = [image_feature_size, max_length]

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
