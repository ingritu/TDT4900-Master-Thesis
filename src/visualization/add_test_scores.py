from pathlib import Path
import json

ROOT_PATH = Path(__file__).absolute().parents[2]


def add_test_scores(df, model, model_dir, dataset):
    with open(model_dir.joinpath('TEST_test_eval.json'), 'r') as json_file:
        eval_scores = json.load(json_file)

    index = len(df)
    df.loc[index, 'model'] = model
    df.loc[index, 'dataset'] = dataset
    df.loc[index, 'b1'] = round(eval_scores['Bleu_1'], 8)
    df.loc[index, 'b2'] = round(eval_scores['Bleu_2'], 8)
    df.loc[index, 'b3'] = round(eval_scores['Bleu_3'], 8)
    df.loc[index, 'b4'] = round(eval_scores['Bleu_4'], 8)
    df.loc[index, 'm'] = round(eval_scores['METEOR'], 8)
    df.loc[index, 'r'] = round(eval_scores['ROUGE_L'], 8)
    df.loc[index, 'c'] = round(eval_scores['CIDEr'], 8)
    df.loc[index, 's'] = round(eval_scores['SPICE'], 8)

    return df
