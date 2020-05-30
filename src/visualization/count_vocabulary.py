from pathlib import Path
import json

ROOT_PATH = Path(__file__).absolute().parents[2]


def count_vocabulary(df, model, model_dir, dataset):
    with open(model_dir.joinpath('TEST_test_result.json'), 'r') as json_file:
        result = json.load(json_file)

    result_corpus = set()
    for obj in result:
        caption = obj['caption']
        # tokenize caption
        caption = caption.lower().split(' ')
        caption = [c.strip() for c in caption]

        # add words to corpus
        result_corpus.update(caption)

    # get size of vocabulary fro test set
    voc_size = len(result_corpus)

    # save to DataFrame
    index = len(df)
    df.loc[index, 'model'] = model
    df.loc[index, 'dataset'] = dataset
    df.loc[index, 'voc_size'] = voc_size

    return df
