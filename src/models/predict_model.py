from pathlib import Path
from src.models.generator_framework import Generator

import argparse

ROOT_PATH = Path(__file__).absolute().parents[2]

if __name__ == '__main__':
    """
    To run script in terminal:
    python3 -m src.models.predict_model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--karpathy', action='store_true',
                        help='Boolean used to decide whether to train on '
                             'the karpathy split of dataset or not.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset to test model on. The options are '
                             '{flickr8k, flickr30k, coco}.')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to evaluate. '
                             'Acceptable values are {train, val, test}.')
    parser.add_argument('--model-name', type=str, default='adaptive_decoder',
                        help='Model type.')
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the models directory. '
                             'Should be something like '
                             'adaptive_decoder_dd-Mon-yyyy_(hh:mm:ss).')
    parser.add_argument('--val-batch-size', type=int, default=250,
                        help='Validation batch size. '
                             'The number of images in a batch. '
                             'The actual batch size is val_batch_size * '
                             'beam_size.')
    parser.add_argument('--beam-size', type=int, default=3,
                        help='Beam size to use in beam search '
                             'inference algorithm. '
                             'Bigger beam size yields higher performance.')
    args = vars(parser.parse_args())

    interim_path = ROOT_PATH.joinpath('data', 'interim')
    processed_path = ROOT_PATH.joinpath('data', 'processed')
    ann_path = processed_path.joinpath('annotations')
    models_path = ROOT_PATH.joinpath('models')
    model_dir = models_path.joinpath(args['model'])
    split_ = args['split']

    assert model_dir.is_dir(), str(model_dir) + " is not a valid directory."
    assert split_ in {'train', 'val', 'test', 'mini_val'}, \
        "Unexpected split. Split must be either train, val or test."

    dataset_ = args['dataset']
    if args['karpathy']:
        interim_path = interim_path.joinpath('karpathy_split')
        ann_path = ann_path.joinpath('karpathy_split')
    test_path = interim_path.joinpath(dataset_ + '_' + split_ + '.csv')
    voc_path_ = interim_path.joinpath(dataset_ + '_vocabulary.csv')
    # modify this later to only load the split to predict on
    feature_path_ = processed_path.joinpath(
        dataset_, 'Images', 'encoded_visual_attention_full.pkl')
    saved_model_path_ = model_dir.joinpath('BEST_checkpoint.pth.tar')

    model_name_ = args['model_name']

    generator = Generator(model_name_, voc_path_, feature_path_)

    generator.load_model(saved_model_path_)

    beam_size_ = args['beam_size']
    val_batch_size = args['val_batch_size']
    print('Using beam Size =', beam_size_, 'and batch size =', val_batch_size)

    res_file = model_dir.joinpath('TEST_' + split_ + '_result.json')
    eval_file = model_dir.joinpath('TEST_' + split_ + '_eval.json')
    if split_ == 'mini_val':
        # did not bother making a mini_val annotation file
        split_ = 'val'
    annFile = ann_path.joinpath(dataset_ + '_' + split_ + '.json')

    # everything is saved to file pluss it is all printed
    result = generator.evaluate(test_path,
                                annFile,
                                res_file,
                                eval_file,
                                batch_size=val_batch_size,
                                beam_size=beam_size_)

    print('CIDEr:', result)
