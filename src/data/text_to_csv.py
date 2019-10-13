
import pandas as pd
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parents[2]

"""
This functionality will be usefull for the Flickr8k and 30k which are 
tokenized but are contained in a txt file, and I would rather want to 
deal with csv files.

This is not meant to be used for the karpathy split. there is a separate 
file for that.
"""


def text_to_csv(file_path, save_path):
    """
    convert info from txt fil to csv.
    """
    # set up caption dictionary
    captions = {}
    labels = ['image_id', 'caption_id', 'caption']
    for l in labels:
        captions[l] = []

    # read txt file an extract info
    with open(file_path, 'r') as file:
        doc = file.read()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(tokens) == 0:
            continue
        # take the first token as image id, the rest as description
        caption_id, caption_tokens = tokens[0], tokens[1:]

        # extract .jpg filename from image id
        image_id = caption_id.split('#')[0]

        # convert description tokens back to string
        caption = ' '.join(caption_tokens)

        # add all info to the caption dictionary
        captions['image_id'].append(image_id)
        captions['caption_id'].append(caption_id)
        captions['caption'].append(caption)

    # convert dict to DataFrame
    cap_df = pd.DataFrame(data=captions, columns=labels)
    cap_df.to_csv(save_path)
