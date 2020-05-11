from pathlib import Path
import argparse
import pandas as pd
import json

ROOT_PATH = Path(__file__).absolute().parents[2]


def add_test_scores(df, model, model_dir, dataset):
    with open(model_dir.joinpath('TEST_test_eval.json'), 'r') as json_file:
        eval_scores = json.load(json_file)

    b1 = round(eval_scores['Bleu_1'], 8)
    b2 = round(eval_scores['Bleu_2'], 8)
    b3 = round(eval_scores['Bleu_3'], 8)
    b4 = round(eval_scores['Bleu_4'], 8)
    m = round(eval_scores['METEOR'], 8)
    r = round(eval_scores['ROUGE_L'], 8)
    c = round(eval_scores['CIDEr'], 8)
    s = round(eval_scores['SPICE'], 8)

    row = pd.Series([model, dataset, b1, b2, b3, b4, m, r, c, s])
    row_df = pd.DataFrame([row])
    df = pd.concat([row_df, df], ignore_index=True)
    return df


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

    file_df.to_csv(data_file)
