from pathlib import Path
import argparse
import pandas as pd

from src.visualization.add_test_scores import add_test_scores

ROOT_PATH = Path(__file__).absolute().parents[2]


if __name__ == '__main__':
    print('Started performance script.')
    """
    python3 -m src.visualization.performance --args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset that the models were trained on.')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='List of models. All models should be trained on '
                             'the same dataset.')
    args = vars(parser.parse_args())

    # print all args
    print("using parsed arguments.")
    for key in args:
        print(key, args[key])

    models_path = ROOT_PATH.joinpath('models')

    data_file = ROOT_PATH.joinpath('data', 'processed', 'test_results.csv')
    if data_file.is_file():
        file_df = pd.read_csv(data_file)
    else:
        labels = ['model', 'dataset',
                  'b1', 'b2', 'b3', 'b4',
                  'm', 'r', 'c', 's']
        file_df = pd.DataFrame(columns=labels)

    for model_ in args['models']:
        model_dir_ = models_path.joinpath(model_)
        file_df = add_test_scores(file_df, model_, model_dir_, args['dataset'])

    file_df.to_csv(data_file, index=False)
