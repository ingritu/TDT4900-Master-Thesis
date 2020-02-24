# TDT4900-Master-Thesis
Image Captioning

## Installations

First go to https://github.com/salaniz/pycocoevalcap to install pycocoevalcap and pycocotools.
We need these packages to evaluate the model, and this is copatible with python 3.

It is possible that you will get an error saying "No module named Cython" 
while you are trying to install the packages. If you get this error just install
Cython by entering this into the terminal:

```
pip install cython
```

Once cython is installed you should be able to install pycocoevalcap and pycocotools.

Then install the rest of the requirements by running:
```
pip install -r requirements.txt
```

## Project Structure
```
/
├── data/
│   ├── raw/
│   │   ├── Flick8k/
│   │   ├── Flickr30k/
│   │   ├── karpathy_split/
│   │   │   ├── datase_coco.json
│   │   │   ├── datase_flickr8k.json
│   │   │   └── datase_flickr30k.json
│   │   └── MSCOCO/
│   │       ├── annotations/
│   │       ├── test2014/
│   │       ├── train2014/
│   │       └── val2014/
│   ├── interim/
│   ├── processed/
│   └── README.md
├── docs/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   ├── annotation_file_exploration.ipynb
│   ├── karpathy_split_exploration.ipynb
│   ├── README.md
│   ├── visualize_word_frequency.ipynb
│   └── visuals/
├── README.md
├── references/
│   ├── annotation_mock.json
│   ├── karpathy_dataset_mock.json
│   ├── logs_and_checkpoints.txt
│   ├── README.md
│   └── run_notes.txt
├── reports/
│   └── README.md
├── requirements.txt
└── src/
    ├── data/
    │   ├── data_cleaning.py
    │   ├── data_generator.py
    │   ├── handle_karpathy_split.py
    │   ├── load_vocabulary.py
    │   ├── make_dataset.py
    │   ├── split_flickr8k.py
    │   ├── subset_splits.py
    │   ├── text_to_csv.py
    │   └── utils.py
    ├── features/
    │   ├── build_features.py
    │   ├── glove_embeddings.py
    │   ├── resize_images.py
    │   └── Resnet_features.py
    ├── __init__.py
    ├── models/
    │   ├── beam.py
    │   ├── custom_layers.py
    │   ├── generator_framework.py
    │   ├── predict_model.py
    │   ├── torch_generators.py
    │   ├── train_model.py
    │   └── utils.py
    └── visualization/
        └── visualize.py
```