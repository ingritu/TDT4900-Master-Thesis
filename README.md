# TDT4900-Master-Thesis
Image Captioning

## Project Structure
```
├── data
│   ├── data_cleaning.py
│   ├── interim
│   ├── processed
│   └── README.md
├── docs
│   └── README.md
├── models
│   └── README.md
├── notebooks
│   └── README.md
├── README.md
├── references
│   └── README.md
├── reports
│   └── README.md
├── requirements.txt
└── src
    ├── data
    │   ├── data_cleaning.py
    │   ├── data_generator.py
    │   ├── handle_karpathy_split.py
    │   ├── karpathy_json_structure.py
    │   ├── load_vocabulary.py
    │   ├── make_dataset.py
    │   ├── max_length_caption.py
    │   ├── split_flickr8k.py
    │   └── text_to_csv.py
    ├── features
    │   ├── build_features.py
    │   ├── glove_embeddings.py
    │   ├── resize_images.py
    │   └── Resnet_features.py
    ├── __init__.py
    ├── models
    │   ├── caption_generator.py
    │   ├── predict_model.py
    │   └── train_model.py
    └── visualization
        └── visualize.py
```