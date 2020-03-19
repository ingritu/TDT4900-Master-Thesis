import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.data.utils import pad_sequences


class TrainCaptionDataset(Dataset):

    def __init__(self, data_path, features, wordtoix, max_len):
        """
        Image captioning dataset.

        Parameters
        ----------
        data_path : Path or str.
        """
        data_path = Path(data_path)
        self.data_df = pd.read_csv(data_path)
        self.features = features
        self.wordtoix = wordtoix
        self.max_len = max_len

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        caption = self.data_df.loc[idx, 'clean_caption']
        image_name = self.data_df.loc[idx, 'image_name']

        caption = [self.wordtoix[word] for word in caption.split(' ')
                   if word in self.wordtoix]
        caplen = len(caption)
        caption = torch.tensor(caption)
        caption = pad_sequences([caption], maxlen=self.max_len).squeeze(0)
        return self.features[image_name], caption, caplen


class TestCaptionDataset(Dataset):

    def __init__(self, data_path, features):
        """
        Image captioning dataset.

        Parameters
        ----------
        data_path : Path or str.
        """
        data_path = Path(data_path)
        self.data_df = pd.read_csv(data_path)
        self.features = features

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        image_name = self.data_df.loc[idx, 'image_name']
        image_id = self.data_df.loc[idx, 'image_id']
        return self.features[image_name], image_id
