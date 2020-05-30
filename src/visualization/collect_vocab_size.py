from pathlib import Path
import argparse
import pandas as pd

from src.visualization.count_vocabulary import count_vocabulary

ROOT_PATH = Path(__file__).absolute().parents[2]

if __name__ == '__main__':
    print('Started collect vocabulary size script.')
    """
    python3 -m src.visualization.collect_vocab_size --args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset that the models were trained on.')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='List of models. All models should be trained on '
                             'the same dataset.')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model type like basic or adaptive.')
    args = vars(parser.parse_args())

    # print all args
    print("using parsed arguments.")
    for key in args:
        print(key, args[key])

    models_path = ROOT_PATH.joinpath('models')

    if args['model_name'] == 'adaptive':
        data_file = ROOT_PATH.joinpath('data',
                                       'processed',
                                       'adaptive_test_vocab_results.csv')
    else:
        data_file = ROOT_PATH.joinpath('data',
                                       'processed',
                                       'test_vocab_results.csv')

    if data_file.is_file():
        file_df = pd.read_csv(data_file)
    else:
        labels = ['model', 'dataset', 'voc_size']
        file_df = pd.DataFrame(columns=labels)

    for model_ in args['models']:
        model_dir_ = models_path.joinpath(model_)
        file_df = count_vocabulary(file_df,
                                   model_,
                                   model_dir_,
                                   args['dataset'])

    file_df.to_csv(data_file, index=False)
