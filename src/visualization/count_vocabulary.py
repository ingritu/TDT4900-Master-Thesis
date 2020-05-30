from pathlib import Path
import json

ROOT_PATH = Path(__file__).absolute().parents[2]


def add_test_scores(df, model, model_dir, dataset):
    with open(model_dir.joinpath('TEST_test_result.json'), 'r') as json_file:
        result = json.load(json_file)



    index = len(df)
    df.loc[index, 'model'] = model
    df.loc[index, 'dataset'] = dataset
    return df