from pathlib import Path
from src.data.utils import max_length_caption
from src.models.generator_framework import Generator

import argparse

ROOT_PATH = Path(__file__).absolute().parents[2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--karpathy', type=bool, default=True,
                        help='Boolean used to decide whether to train on '
                             'the karpathy split of dataset or not.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset to test model on. The options are '
                             '{flickr8k, flickr30k, coco}.')
    parser.add_argument('--model_name', type=str, default='adaptive_decoder',
                        help='Model type.')
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the models directory. '
                             'Should be something like '
                             'adaptive_decoder_dd-Mon-yyyy_(hh:mm:ss).')
    parser.add_argument('--val_batch_size', type=int, default=250,
                        help='Validation batch size. '
                             'The number of images in a batch. '
                             'The actual batch size is val_batch_size * '
                             'beam_size.')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='Beam size to use in beam search '
                             'inference algorithm. '
                             'Bigger beam size yields higher performance.')
    args = vars(parser.parse_args())

    interim_path = ROOT_PATH.joinpath('data', 'interim')
    processed_path = ROOT_PATH.joinpath('data', 'processed')
    models_path = ROOT_PATH.joinpath('models')
    model_dir = models_path.joinpath(args['model'])

    dataset_ = args['dataset']
    if args['karpathy']:
        interim_path = interim_path.joinpath('karpathy_split')
    train_path = interim_path.joinpath(dataset_, dataset_ + '_train_clean.csv')
    val_path = interim_path.joinpath(dataset_, dataset_ + '_mini_test.csv')
    voc_path_ = interim_path.joinpath(dataset_, dataset_ + '_vocabulary.csv')
    feature_path_ = processed_path.joinpath(
        dataset_, 'Images', 'encoded_visual_attention_full.pkl')
    saved_model_path_ = model_dir.joinpath('BEST_checkpoint.pth.tar')

    model_name_ = args['model_name']

    # bull values that generator needs to be set
    save_path_ = models_path
    em_dim = 300
    hidden_shape_ = 50
    loss_function_ = 'cross_entropy'
    opt = 'adam'
    lr_ = 0.0001
    seed_ = 222

    # actual useful values
    max_length = max_length_caption(train_path)
    input_shape_ = [[8, 8, 1536], max_length]

    generator = Generator(model_name_, input_shape_, hidden_shape_,
                          voc_path_, feature_path_,
                          save_path_,
                          loss_function=loss_function_,
                          optimizer=opt, lr=lr_,
                          embedding_size=em_dim,
                          seed=seed_)

    generator.load_model(saved_model_path_)

    print('Making predictions!')
    beam_size_ = args['beam_size']
    val_batch_size = args['val_batch_size']
    print('Using beam Size =', beam_size_, 'and batch size =', val_batch_size)
    



    result = generator.evaluate(val_path,
                                batch_size=val_batch_size,
                                beam_size=beam_size_)

    print(result)




