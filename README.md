# TDT4900-Master-Thesis
Image Captioning

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
│   ├── karpathy_split_exploration.ipynb
│   ├── README.md
│   ├── visualize_word_frequency.ipynb
│   └── visuals/
├── README.md
├── references/
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
    │   ├── max_length_caption.py
    │   ├── split_flickr8k.py
    │   ├── subset_splits.py
    │   └── text_to_csv.py
    ├── features/
    │   ├── build_features.py
    │   ├── glove_embeddings.py
    │   ├── resize_images.py
    │   └── Resnet_features.py
    ├── __init__.py
    ├── models/
    │   ├── custom_layers.py
    │   ├── generator_framework.py
    │   ├── predict_model.py
    │   ├── torch_generators.py
    │   └── train_model.py
    └── visualization/
        └── visualize.py
```