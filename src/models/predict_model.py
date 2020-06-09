from pathlib import Path
from src.models.generator_framework import Generator
from src.visualization.add_test_scores import add_test_scores

import sys
import torch
import torch.multiprocessing as mp
import numpy as np
from src.utils import get_gpu_name
from src.utils import get_cuda_version
from src.utils import get_cudnn_version

import argparse
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]

if __name__ == '__main__':
    """
    To run script in terminal:
    python3 -m src.models.predict_model
    """
    print("Started predict model script.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--karpathy', action='store_true',
                        help='Boolean used to decide whether to train on '
                             'the karpathy split of dataset or not.')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset to test model on. The options are '
                             '{flickr8k, flickr30k, coco}. '
                             'The default value is "coco".')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to evaluate. '
                             'Acceptable values are {train, val, test}. '
                             'The default value is "val".')
    parser.add_argument('--model-name', type=str, default='adaptive',
                        help='Model type. '
                             'The default value is adaptive.')
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the models directory. '
                             'Should be something like '
                             'adaptive_decoder_dd-Mon-yyyy_(hh:mm:ss).')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='Validation batch size. '
                             'The number of images in a batch. '
                             'The actual batch size is val_batch_size * '
                             'beam_size. '
                             'The default value is 1.')  # do not change this value
    parser.add_argument('--beam-size', type=int, default=3,
                        help='Beam size to use in beam search '
                             'inference algorithm. '
                             'Bigger beam size yields higher performance. '
                             'The default value is 3.')
    parser.add_argument('--update-results-file', action='store_true',
                        help='Predict model will automatically update the '
                             'test_results file if the test set is test. '
                             'Use this flag to turn on that feature.')
    args = vars(parser.parse_args())

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
    # vocabulary for regular coco karpathy split is in interim not processed
    main_data_path = interim_path if dataset_ == 'coco' else processed_path

    if args['karpathy']:
        main_data_path = main_data_path.joinpath('karpathy_split')
        ann_path = ann_path.joinpath('karpathy_split')

    test_path = main_data_path.joinpath('coco_' + split_ + '.csv')
    voc_path_ = main_data_path.joinpath(dataset_ + '_vocabulary.csv')
    # modify this later to only load the split to predict on
    feature_path_ = processed_path.joinpath(
        'images', 'karpathy_split',
        'coco_encoded_visual_attention_full.pkl')
    saved_model_path_ = model_dir.joinpath('BEST_checkpoint.pth.tar')

    model_name_ = args['model_name']

    print('vocabulary:', voc_path_)
    print('features:', feature_path_)
    print('dataset;', test_path)

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
    annFile = ann_path.joinpath('coco_' + split_ + '.json')

    # everything is saved to file plus it is all printed
    _ = generator.evaluate(test_path,
                           annFile,
                           res_file,
                           eval_file,
                           batch_size=val_batch_size,
                           beam_size=beam_size_)

    if split_ == 'test' and args['update_results_file']:
        # update results file automatically
        print("Update results file!!!")
        if model_name_ == 'adaptive':
            data_file = ROOT_PATH.joinpath('data',
                                           'processed',
                                           'adaptive_test_results.csv')
        else:
            data_file = ROOT_PATH.joinpath('data',
                                           'processed',
                                           'test_results.csv')

        if data_file.is_file():
            file_df = pd.read_csv(data_file)
        else:
            labels = ['model', 'dataset',
                      'b1', 'b2', 'b3', 'b4',
                      'm', 'r', 'c', 's']
            file_df = pd.DataFrame(columns=labels)
        add_test_scores(file_df, args['model'], model_dir, dataset_)
        file_df.to_csv(data_file, index=False)
