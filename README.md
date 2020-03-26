# TDT4900-Master-Thesis
Image Captioning

## Installations

First go to https://github.com/salaniz/pycocoevalcap to install pycocoevalcap and pycocotools.
We need these packages to evaluate the model, and this is compatible with python 3.

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

## Running project
### 1. Make the dataset
```
python3 -m src.data.make_dataset --help

usage: make_dataset.py [-h] [--dataset DATASET] [--karpathy]
                       [--threshold THRESHOLD]
                       [--unk-percentage UNK_PERCENTAGE]
                       [--cutoff-value CUTOFF_VALUE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset to train on. The options are {flickr8k,
                        flickr30k, coco}. The default value is "coco".
  --karpathy            Boolean used to decide whether to train on the
                        karpathy split of dataset or not.
  --threshold THRESHOLD
                        Minimum word frequency for words included in the
                        vocabulary. The defualt value is 5.
  --unk-percentage UNK_PERCENTAGE
                        The percentage of UNK tokens in a caption must be
                        below this value in order to be included in the train
                        set. The default value is 0.3.
  --cutoff-value CUTOFF_VALUE
                        As a part of the pre-processing we will augment
                        captions that are considered too long. This argument
                        essentially sets the max length of a caption,
                        excluding the startseq and endseq tokens. The default
                        value is 16.
```
### 2. Resize the images
```
usage: resize_images.py [-h] --new-image-size NEW_IMAGE_SIZE
                        [NEW_IMAGE_SIZE ...] [--image-split IMAGE_SPLIT]
                        [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --new-image-size NEW_IMAGE_SIZE [NEW_IMAGE_SIZE ...]
                        List new image dimensions. should be something like
                        299 299.
  --image-split IMAGE_SPLIT
                        Which dataset split images to resize. Default is full,
                        meaning all images in the dataset will be resized.
                        This is only necessary for coco since it is so big.
                        The default value is "full".
  --dataset DATASET     Which dataset to resize its images. The options are
                        {flickr8k, flickr30k, coco}. The default value is
                        "coco".
```
### 3. Make the features
```
python3 -m src.features.build_features --help

usage: build_features.py [-h] --new-image-size NEW_IMAGE_SIZE
                         [NEW_IMAGE_SIZE ...] [--feature-split FEATURE_SPLIT]
                         [--karpathy] [--visual-attention]
                         [--output-layer-idx OUTPUT_LAYER_IDX]
                         [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --new-image-size NEW_IMAGE_SIZE [NEW_IMAGE_SIZE ...]
                        List new image dimensions. should be something like
                        299 299.
  --feature-split FEATURE_SPLIT
                        Which dataset split to make features for. Default
                        value is full, meaning all images in the dataset will
                        be encoded and saved in the same file. The default
                        value is "full".
  --karpathy            Boolean used to decide whether to train on the
                        karpathy split of dataset or not.
  --visual-attention    Boolean for deciding whether to extract visual
                        features that are usable for models that use visual
                        attention.
  --output-layer-idx OUTPUT_LAYER_IDX
                        Which layer to extract features from. The default
                        value is -3.
  --dataset DATASET     Which dataset to create image features for. The
                        options are {flickr8k, flickr30k, coco}. The default
                        value "coco".
```

### 4. Train a model
```
python3 -m src.models.train_model --help

usage: train_model.py [-h] [--batch-size BATCH_SIZE] [--beam-size BEAM_SIZE]
                      [--val-batch-size VAL_BATCH_SIZE] [--epochs EPOCHS]
                      [--early-stopping-freq EARLY_STOPPING_FREQ]
                      [--val-metric VAL_METRIC] [--not-validate]
                      [--embedding-size EMBEDDING_SIZE]
                      [--hidden-size HIDDEN_SIZE]
                      [--loss-function LOSS_FUNCTION] [--optimizer OPTIMIZER]
                      [--lr LR] [--seed SEED] [--model MODEL]
                      [--dropout DROPOUT] [--karpathy] [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Training batch size. The number of captions in a
                        batch. The default value is 80.
  --beam-size BEAM_SIZE
                        Beam size to use in beam search inference algorithm.
                        Bigger beam size yields higher performance. The
                        default value is 3.
  --val-batch-size VAL_BATCH_SIZE
                        Validation batch size. The number of images in a
                        batch. The actual batch size is val_batch_size *
                        beam_size. The default value is 250.
  --epochs EPOCHS       The number of epochs to train the network for. The
                        default value is 50 epochs.
  --early-stopping-freq EARLY_STOPPING_FREQ
                        Training will stop if no improvements have been made
                        over this many epochs. The default value is 6.
  --val-metric VAL_METRIC
                        Automatic evaluation metric to consider for
                        validation. Acceptable values are {Bleu_1, Bleu_2,
                        Bleu_3, Bleu_4, ROUGE_L, METEOR, CIDEr, SPICE}. The
                        default value is CIDEr.
  --not-validate        Bool for switching on and off COCO evaluation.
                        Activating flag means to not do COCO evaluation.
  --embedding-size EMBEDDING_SIZE
                        Embedding dimension. The size of the word vector
                        representations. The default value is 512.
  --hidden-size HIDDEN_SIZE
                        Hidden dimension. The default value is 512.
  --loss-function LOSS_FUNCTION
                        Loss/Cost function to use during training. The default
                        value is cross_entropy.
  --optimizer OPTIMIZER
                        Optimizer to use during training. The default value is
                        adam.
  --lr LR               Initial learning rate for the decoder. The default
                        value is 0.001.
  --seed SEED           Random state seed.
  --model MODEL         Model name. Which model type to train. The default
                        value is "adaptive".
  --dropout DROPOUT     Use dropout on some layers. Decide the dropout value.
                        The default value is 0.5
  --karpathy            Boolean used to decide whether to train on the
                        karpathy split of dataset or not.
  --dataset DATASET     Dataset to train on. The options are {flickr8k,
                        flickr30k, coco}. The default value is "coco".
```
### 5. Evaluate a trained model
```
python3 -m src.models.predict_model --help

usage: predict_model.py [-h] [--karpathy] [--dataset DATASET] [--split SPLIT]
                        [--model-name MODEL_NAME] --model MODEL
                        [--val-batch-size VAL_BATCH_SIZE]
                        [--beam-size BEAM_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --karpathy            Boolean used to decide whether to train on the
                        karpathy split of dataset or not.
  --dataset DATASET     Dataset to test model on. The options are {flickr8k,
                        flickr30k, coco}. The default value is "coco".
  --split SPLIT         Dataset split to evaluate. Acceptable values are
                        {train, val, test}. The default value is "val".
  --model-name MODEL_NAME
                        Model type. The default value is adaptive.
  --model MODEL         Name of the models directory. Should be something like
                        adaptive_decoder_dd-Mon-yyyy_(hh:mm:ss).
  --val-batch-size VAL_BATCH_SIZE
                        Validation batch size. The number of images in a
                        batch. The actual batch size is val_batch_size *
                        beam_size. The default value is 250.
  --beam-size BEAM_SIZE
                        Beam size to use in beam search inference algorithm.
                        Bigger beam size yields higher performance. The
                        default value is 3.
```
## Project Structure
This project uses the Data Science CookieCutter template.
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
│   │   ├── annotations/
│   │   ├── cap_subsets/
│   │   │   ├── c1.csv
│   │   │   ├── c2.csv
│   │   │   ├── c3.csv
│   │   │   ├── c4.csv
│   │   │   └── c5.csv
│   │   └── images/
│   │       └── karpathy_split/
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
## Acknowledgements
The model implementation is based on the work of 
https://github.com/jiasenlu/AdaptiveAttention (original adaptive model), 
https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention (pytorch implementation) and 
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning (pytorch implementation of Show, Attend and Tell).