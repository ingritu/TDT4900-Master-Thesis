import argparse
from pathlib import Path
import pandas as pd

from src.visualization.sample_image_captions import sample_image_captions

ROOT_PATH = Path(__file__).absolute().parents[2]

if __name__ == '__main__':
    print('Started collect caption samples script.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='List of models to collect '
                             'generated captions from.')
    parser.add_argument('--sample-size', type=int, default=20,
                        help='Number of images to sample from test set.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed to determine which images get '
                             'picked as part of the sample set.')

    args = vars(parser.parse_args())

    print('Parsed Arguments')
    for k, v in args.items():
        print(k, v)

    seed = args['seed']
    models = args['models']
    sample_size = args['sample_size']

    processed_path = ROOT_PATH.joinpath('data', 'processed')
    test_path = processed_path.joinpath('coco_test.csv')
    save_name = 'TEST_sample_' + str(seed) + '.csv'
    save_file = processed_path.joinpath(save_name)

    if save_file.is_file():
        # file already exist and we want to add more model results to it
        res_df = pd.read_csv(save_file)
    else:
        # file does not exist
        labels = ['model_name', 'model_type', 'dataset',
                  'image_id', 'image_name', 'caption']
        res_df = pd.DataFrame(columns=labels)

    res_df = sample_image_captions(models,
                                   test_path,
                                   res_df,
                                   seed,
                                   sample_size=sample_size)

    # save results
    res_df.to_csv(save_file, index=False)
