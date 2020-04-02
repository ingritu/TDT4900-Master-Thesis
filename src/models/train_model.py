from pathlib import Path
from src.models.generator_framework import Generator
from src.utils import get_gpu_name
from src.utils import get_cuda_version
from src.utils import get_cudnn_version
import sys
import argparse


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
                             'The number of captions in a batch. '
                             'The default value is 80.')
    parser.add_argument('--beam-size', type=int, default=3,
                        help='Beam size to use in beam search '
                             'inference algorithm. '
                             'Bigger beam size yields higher performance. '
                             'The default value is 3.')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='Validation batch size. The number of images in a'
                             ' batch. The actual batch size is '
                             'val_batch_size * beam_size. '
                             'The default value is 1.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='The number of epochs to train the network for. '
                             'The default value is 50 epochs.')
    parser.add_argument('--early-stopping-freq', type=int, default=6,
                        help='Training will stop if no improvements have been '
                             'made over this many epochs. '
                             'The default value is 6.')
    parser.add_argument('--val-metric', type=str, default='CIDEr',
                        help='Automatic evaluation metric to consider for '
                             'validation. Acceptable values are {Bleu_1, '
                             'Bleu_2, Bleu_3, Bleu_4, ROUGE_L, METEOR, '
                             'CIDEr, SPICE}. '
                             'The default value is CIDEr.')
    parser.add_argument('--not-validate', action='store_true',
                        help='Bool for switching on and off COCO evaluation. '
                             'Activating flag means to not do '
                             'COCO evaluation.')
    parser.add_argument('--lr-decay-start', type=int, default=20,
                        help='when to start decaying the learning rate. '
                             'The default value is 20.')
    parser.add_argument('--lr-decay-every', type=int, default=5,
                        help='how often to decay the learning rate. '
                             'The default value is 5.')
    parser.add_argument('--lr-decay-factor', type=float, default=0.5,
                        help='Factor to decay lr with. '
                             'The default value is 0.5.')
    parser.add_argument('--clip-value', type=float, default=0.1,
                        help='Value to clip gradients by. '
                             'The default value is 0.1.')
    # Model details
    parser.add_argument('--embedding-size', type=int, default=512,
                        help='Embedding dimension. '
                             'The size of the word vector representations. '
                             'The default value is 512.')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Hidden dimension. '
                             'The default value is 512.')
    parser.add_argument('--loss-function', type=str, default='cross_entropy',
                        help='Loss/Cost function to use during training. '
                             'The default value is cross_entropy.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use during training. '
                             'The default value is adam.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate for the decoder. '
                             'The default value is 0.001.')
    parser.add_argument('--seed', type=int, default=222,
                        help='Random state seed.')
    parser.add_argument('--model', type=str, default='adaptive',
                        help='Model name. Which model type to train. '
                             'The default value is "adaptive".')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Use dropout on some layers. '
                             'Decide the dropout value. '
                             'The default value is 0.5')
    # data details
    parser.add_argument('--karpathy', action='store_true',
                        help='Boolean used to decide whether to train on '
                             'the karpathy split of dataset or not.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset to train on. The options are '
                             '{flickr8k, flickr30k, coco}. '
                             'The default value is "coco".')
    parser.add_argument('--mini', action='store_true',
                        help='switch for using custom mini sets.')
    # there still are more customizable parameters to set,
    # add these later
    args = vars(parser.parse_args())  # access args as dictionary
    # SEEDING TRAINING
    seed_ = args['seed']
    torch.manual_seed(seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_)

    print("OS: ", sys.platform)
    print("Python: ", sys.version)
    print("PyTorch: ", torch.__version__)
    print("Numpy: ", np.__version__)
    print("CUDA:", get_cuda_version())
    print("CuDNN:", get_cudnn_version())

    num_gpus = torch.cuda.device_count()
    num_cpus = mp.cpu_count()
    multi_gpus = num_gpus > 1
    print("GPUs:", num_gpus)
    print("CPUs:", num_cpus)
    print("GPU:", get_gpu_name())
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

    mini_ = args['mini']
    annFile = ann_path.joinpath(dataset + '_val.json')
    if mini_:
        train_path = interim_path.joinpath(dataset + '_mini_train_clean.csv')
        val_path = interim_path.joinpath(dataset + '_mini_val.csv')
    else:
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
    not_validate_ = args['not_validate']
    lr_decay_start_ = args['lr_decay_start']
    lr_decay_every_ = args['lr_decay_every']
    lr_decay_factor_ = args['lr_decay_factor']
    clip_value_ = args['clip_value']
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
                    validation_metric=val_metric,
                    not_validate=not_validate_,
                    lr_decay_start=lr_decay_start_,
                    lr_decay_every=lr_decay_every_,
                    lr_decay_factor=lr_decay_factor_,
                    clip_value=clip_value_)
    print("Finished training model!")
