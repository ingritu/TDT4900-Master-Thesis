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

## Running project
### 1. Make the dataset
```
python3 -m src.data.make_dataset --help

usage: make_dataset.py [-h] [--dataset DATASET] [--karpathy]
                       [--threshold THRESHOLD]
                       [--unk_percentage UNK_PERCENTAGE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset to train on. The options are {flickr8k,
                        flickr30k, coco}.
  --karpathy            Boolean used to decide whether to train on the
                        karpathy split of dataset or not.
  --threshold THRESHOLD
                        Minimum word frequency for words included in the
                        vocabulary.
  --unk-percentage UNK_PERCENTAGE
                        The percentage of UNK tokens in a caption must be
                        below this value in order to be included in the train
                        set.
```
### 2. Make the features
```
python3 -m src.features.build_features --help

usage: build_features.py [-h] [--resize-images]
                         [--new-image-size NEW_IMAGE_SIZE [NEW_IMAGE_SIZE ...]]
                         [--dataset DATASET] [--visual-attention]
                         [--output-layer-idx OUTPUT_LAYER_IDX]

optional arguments:
  -h, --help            show this help message and exit
  --resize-images       Boolean to decide whether to resize the images before
                        building the actual features.
  --new-image-size NEW_IMAGE_SIZE [NEW_IMAGE_SIZE ...]
                        List new image dimensions. should be something like
                        299 299.
  --dataset DATASET     Which dataset to create image features for. The
                        options are {flickr8k, flickr30k, coco}.
  --visual-attention    Boolean for deciding whether to extract visual
                        features that are usable for models that use visual
                        attention.
  --output-layer-idx OUTPUT_LAYER_IDX
                        Which layer to extract features from.
```

### 3. Train a model
```
python3 -m src.models.train_model --help

usage: train_model.py [-h] [--batch-size BATCH_SIZE]
                      [--val-batch_size VAL_BATCH_SIZE]
                      [--beam-size BEAM_SIZE] [--epochs EPOCHS]
                      [--early-stopping_freq EARLY_STOPPING_FREQ]
                      [--val-metric VAL_METRIC]
                      [--embedding-size EMBEDDING_SIZE]
                      [--hidden-size HIDDEN_SIZE] [--num-lstms NUM_LSTMS]
                      [--decoding-stack-size DECODING_STACK_SIZE]
                      [--loss-function LOSS_FUNCTION] [--optimizer OPTIMIZER]
                      [--lr LR] [--seed SEED] [--model MODEL] [--karpathy]
                      [--dataset DATASET] --image-feature-size
                      IMAGE_FEATURE_SIZE [IMAGE_FEATURE_SIZE ...]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Training batch size. The number of captions in a
                        batch.
  --val-batch-size VAL_BATCH_SIZE
                        Validation batch size. The number of images in a
                        batch. The actual batch size is val_batch_size *
                        beam_size.
  --beam-size BEAM_SIZE
                        Beam size to use in beam search inference algorithm.
                        Bigger beam size yields higher performance.
  --epochs EPOCHS       The number of epochs to train the network for.
  --early-stopping_freq EARLY_STOPPING_FREQ
                        Training will stop if no improvements have been made
                        over this many epochs. Default value is 6.
  --val-metric VAL_METRIC
                        Automatic evaluation metric to consider for
                        validation. Acceptable values are {Bleu_1, Bleu_2,
                        Bleu_3, Bleu_4, ROUGE_L, METEOR, CIDEr, SPICE}. The
                        default value is CIDEr.
  --embedding-size EMBEDDING_SIZE
                        Embedding dimension. The size of the word vector
                        representations.
  --hidden-size HIDDEN_SIZE
                        Hidden dimension.
  --num-lstms NUM_LSTMS
                        The number of LSTM cells to stack. Default value is 1.
  --decoding-stack-size DECODING_STACK_SIZE
                        The number of Linear layers to stack in the multimodal
                        decoding part of the model.
  --loss-function LOSS_FUNCTION
                        Loss/Cost function to use during training.
  --optimizer OPTIMIZER
                        Optimizer to use during training.
  --lr LR               Initial learning rate for the decoder.
  --seed SEED           Random state seed.
  --model MODEL         Model name. Which model type to train.
  --karpathy            Boolean used to decide whether to train on the
                        karpathy split of dataset or not.
  --dataset DATASET     Dataset to train on. The options are {flickr8k,
                        flickr30k, coco}.
  --image-feature-size IMAGE_FEATURE_SIZE [IMAGE_FEATURE_SIZE ...]
                        List integers. Should be something like
                        --image_feature_size 8 8 1536.
```
### 4. Evaluate a trained model
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
                        flickr30k, coco}.
  --split SPLIT         Dataset split to evaluate. Acceptable values are
                        {train, val, test}.
  --model-name MODEL_NAME
                        Model type.
  --model MODEL         Name of the models directory. Should be something like
                        adaptive_decoder_dd-Mon-yyyy_(hh:mm:ss).
  --val-batch-size VAL_BATCH_SIZE
                        Validation batch size. The number of images in a
                        batch. The actual batch size is val_batch_size *
                        beam_size.
  --beam-size BEAM_SIZE
                        Beam size to use in beam search inference algorithm.
                        Bigger beam size yields higher performance.

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