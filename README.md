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
python3 -m src.features.resize_images --help

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
                      [--lr-decay-start LR_DECAY_START]
                      [--lr-decay-every LR_DECAY_EVERY]
                      [--lr-decay-factor LR_DECAY_FACTOR]
                      [--clip-value CLIP_VALUE]
                      [--embedding-size EMBEDDING_SIZE]
                      [--hidden-size HIDDEN_SIZE]
                      [--loss-function LOSS_FUNCTION] [--optimizer OPTIMIZER]
                      [--lr LR] [--seed SEED] [--model MODEL]
                      [--dropout DROPOUT] [--karpathy] [--dataset DATASET]
                      [--mini]

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
                        beam_size. The default value is 1.
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
  --lr-decay-start LR_DECAY_START
                        when to start decaying the learning rate. The default
                        value is 20.
  --lr-decay-every LR_DECAY_EVERY
                        how often to decay the learning rate. The default
                        value is 5.
  --lr-decay-factor LR_DECAY_FACTOR
                        Factor to decay lr with. The default value is 0.5.
  --clip-value CLIP_VALUE
                        Value to clip gradients by. The default value is 0.1.
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
  --mini                switch for using custom mini sets.
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
## Running Experiments
### Make Experiments
```
python3 -m src.data.make_experiments
```
### Run experiment
```
python3 -m src.models.do_experiments --help

usage: do_experiments.py [-h] [--batch-size BATCH_SIZE]
                         [--beam-size BEAM_SIZE]
                         [--val-batch-size VAL_BATCH_SIZE] [--epochs EPOCHS]
                         [--early-stopping-freq EARLY_STOPPING_FREQ]
                         [--val-metric VAL_METRIC] [--not-validate]
                         [--lr-decay-start LR_DECAY_START]
                         [--lr-decay-every LR_DECAY_EVERY]
                         [--lr-decay-factor LR_DECAY_FACTOR]
                         [--clip-value CLIP_VALUE]
                         [--embedding-size EMBEDDING_SIZE]
                         [--hidden-size HIDDEN_SIZE]
                         [--loss-function LOSS_FUNCTION]
                         [--optimizer OPTIMIZER] [--lr LR] [--seed SEED]
                         [--model MODEL] [--dropout DROPOUT]
                         [--dataset DATASET]

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
                        beam_size. The default value is 1.
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
  --lr-decay-start LR_DECAY_START
                        when to start decaying the learning rate. The default
                        value is 20.
  --lr-decay-every LR_DECAY_EVERY
                        how often to decay the learning rate. The default
                        value is 5.
  --lr-decay-factor LR_DECAY_FACTOR
                        Factor to decay lr with. The default value is 0.5.
  --clip-value CLIP_VALUE
                        Value to clip gradients by. The default value is 0.1.
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
  --dataset DATASET     Experiment Dataset to train on. The options are {c1,
                        c2, c3, c4, c5, cp6, cp7, cp8, cp9, cp10, c1p1, c2p2,
                        c3p3, c4p4}. The default value is "coco".
```

## Project Structure
This project uses the Data Science CookieCutter template. Some of the scripts 
ended up not being used for the pipeline but they have not been removed since 
they were a part of the process.
```
/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
├── docs/
│   └── README.md
├── doxygen_config
├── models/
│   └── README.md
├── notebooks/
│   └── README.md
├── README.md
├── references/
│   └── README.md
├── reports/
│   └── README.md
├── requirements.txt
└── src/
    ├── data/
    │   ├── combine_files.py
    │   ├── data_cleaning.py
    │   ├── data_generator.py
    │   ├── dataset.py
    │   ├── handle_karpathy_split.py
    │   ├── make_dataset.py
    │   ├── make_experiments.py
    │   ├── prepare_for_translate.py
    │   ├── preprocess_coco.py
    │   ├── reduce_images_in_dataset.py
    │   ├── repair_files.py
    │   ├── split_flickr8k.py
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
    │   ├── do_experiments.py
    │   ├── generator_framework.py
    │   ├── predict_model.py
    │   ├── torch_generators.py
    │   ├── train_model.py
    │   └── utils.py
    ├── visualization/
    │   ├── add_test_scores.py
    │   ├── collect_caption_samples.py
    │   ├── collect_test_scores.py
    │   ├── collect_vocab_size.py
    │   ├── count_vocabulary.py
    │   ├── sample_image_captions.py
    │   └── visualize.py
    └── utils.py
```
## Acknowledgements
The model implementation is based on the work of 
https://github.com/jiasenlu/AdaptiveAttention (original adaptive model), 
https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention (pytorch implementation) and 
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning (pytorch implementation of Show, Attend and Tell).