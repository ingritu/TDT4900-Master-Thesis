from pathlib import Path
from src.models.generator_framework import Generator
import sys
import argparse

# import to set seed in this program
import torch
import torch.multiprocessing as mp
import numpy as np

ROOT_PATH = Path(__file__).absolute().parents[2]


if __name__ == '__main__':
    """
    To run script in terminal:
    python3 -m src.models.train_model --args
    """
    print("Started train model script.")
    # All default values are the values used in the knowing when to look paper
    parser = argparse.ArgumentParser()
    # Training details
    parser.add_argument('--batch-size', type=int, default=80,
                        help='Training batch size. '
                             'The number of captions in a batch.')
    parser.add_argument('--beam-size', type=int, default=3,
                        help='Beam size to use in beam search '
                             'inference algorithm. '
                             'Bigger beam size yields higher performance.')
    parser.add_argument('--val-batch-size', type=int, default=250,
                        help='Validation batch size. The number of images in a'
                             ' batch. The actual batch size is '
                             'val_batch_size * beam_size.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='The number of epochs to train the network for.')
    parser.add_argument('--early-stopping-freq', type=int, default=6,
                        help='Training will stop if no improvements have been '
                             'made over this many epochs. Default value is 6.')
    parser.add_argument('--val-metric', type=str, default='CIDEr',
                        help='Automatic evaluation metric to consider for '
                             'validation. Acceptable values are {Bleu_1, '
                             'Bleu_2, Bleu_3, Bleu_4, ROUGE_L, METEOR, '
                             'CIDEr, SPICE}. The default value is CIDEr.')
    # Model details
    parser.add_argument('--embedding-size', type=int, default=512,
                        help='Embedding dimension. '
                             'The size of the word vector representations.')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Hidden dimension.')
    parser.add_argument('--loss-function', type=str, default='cross_entropy',
                        help='Loss/Cost function to use during training.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use during training.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate for the decoder.')
    parser.add_argument('--seed', type=int, default=222,
                        help='Random state seed.')
    parser.add_argument('--model', type=str, default='adaptive',
                        help='Model name. Which model type to train.')
    # data details
    parser.add_argument('--karpathy', action='store_true',
                        help='Boolean used to decide whether to train on '
                             'the karpathy split of dataset or not.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset to train on. The options are '
                             '{flickr8k, flickr30k, coco}.')
    # there still are more customizable parameters to set,
    # add these later
    args = vars(parser.parse_args())  # access args as dictionary
    # SEEDING TRAINING
    seed_ = args['seed']
    torch.manual_seed(seed_)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed_)

    print("OS: ", sys.platform)
    print("Python: ", sys.version)
    print("PyTorch: ", torch.__version__)
    print("Numpy: ", np.__version__)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    num_cpus = mp.cpu_count()
    multi_gpus = num_gpus > 1
    print("GPUs:", num_gpus)
    print("CPUs:", num_cpus)
    # print all args
    print("using parsed arguments.")
    for key in args:
        print(key, args[key])

    interim_path = ROOT_PATH.joinpath('data',
                                      'interim')
    processed_path = ROOT_PATH.joinpath('data',
                                        'processed')
    ann_path = processed_path.joinpath('annotations')
    feature_path = processed_path.joinpath('images')
    dataset = args['dataset']
    if args['karpathy']:
        interim_path = interim_path.joinpath('karpathy_split')
        # annotation file
        ann_path = ann_path.joinpath('karpathy_split')
        feature_path = feature_path.joinpath('karpathy_split')

    annFile = ann_path.joinpath(dataset + '_val.json')
    train_path = interim_path.joinpath(dataset + '_train_clean.csv')
    val_path = interim_path.joinpath(dataset + '_val.csv')
    voc_path_ = interim_path.joinpath(dataset + '_vocabulary.csv')
    featureFile = feature_path.joinpath(dataset +
                                        '_encoded_visual_attention_full.pkl')

    save_path_ = ROOT_PATH.joinpath('models')

    # training
    batch_size = args['batch_size']
    beam_size = args['beam_size']
    val_batch_size = args['val_batch_size']
    epochs = args['epochs']
    early_stopping_freq = args['early_stopping_freq']
    val_metric = args['val_metric']
    # model
    model_name_ = args['model']
    em_dim = args['embedding_size']
    hidden_size_ = args['hidden_size']
    loss_function_ = args['loss_function']
    opt = args['optimizer']
    lr_ = args['lr']

    if multi_gpus:
        lr_ *= num_gpus
        batch_size *= num_gpus

    generator = Generator(model_name_,
                          voc_path_,
                          featureFile)
    generator.compile(save_path_,
                      embedding_size=em_dim,
                      hidden_size=hidden_size_,
                      loss_function=loss_function_,
                      optimizer=opt,
                      lr=lr_,
                      multi_gpus=multi_gpus)

    # model is automatically saved after training
    generator.train(train_path,
                    val_path,
                    annFile,
                    epochs=epochs,
                    batch_size=batch_size,
                    early_stopping_freq=early_stopping_freq,
                    val_batch_size=val_batch_size,
                    beam_size=beam_size,
                    validation_metric=val_metric)
    print("Finished training model!")
