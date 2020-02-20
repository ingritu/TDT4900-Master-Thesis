import torch
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence
from pathlib import Path
import pandas as pd
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
import numpy as np
from copy import deepcopy
from time import time
import json

from src.data.data_generator import data_generator
from src.data.data_generator import pad_sequences
from src.data.load_vocabulary import load_vocabulary
from src.features.Resnet_features import load_visual_features
from src.models.torch_generators import model_switcher
from src.models.utils import save_checkpoint
from src.models.utils import save_training_log

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

ROOT_PATH = Path(__file__).absolute().parents[2]


def loss_switcher(loss_string):
    loss_string = loss_string.lower()
    switcher = {
        'cross_entropy': nn.CrossEntropyLoss,
        'mse': nn.MSELoss,
        'default': nn.CrossEntropyLoss,
    }

    return switcher[loss_string]


def optimizer_switcher(optimizer_string):
    optimizer_string = optimizer_string.lower()
    switcher = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'default': optim.SGD
    }
    return switcher[optimizer_string]


class Generator:

    def __init__(self,
                 model_name,
                 input_shape,
                 hidden_size,
                 voc_path,
                 feature_path,
                 save_path,
                 loss_function='cross_entropy',
                 optimizer='adam',
                 lr=0.0001,
                 embedding_size=300,
                 seed=222):
        # delete if not used in this class
        self.input_shape = input_shape
        self.max_length = self.input_shape[1]

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.save_path = save_path
        self.random_seed = seed

        self.wordtoix, self.ixtoword = load_vocabulary(voc_path)
        self.vocab_path = voc_path
        self.vocab_size = len(self.wordtoix)
        self.feature_path = feature_path
        self.encoded_features = load_visual_features(feature_path)

        # initialize model as None
        self.model = None
        self.train_params = 0

        self.model_name = model_name

        # initialize loss function
        self.loss_string = loss_function
        self.criterion = loss_switcher(self.loss_string)()

        # set up optimizer
        self.optimizer_string = optimizer
        self.optimizer = None  # not initialized
        self.lr = lr

        # misc
        self.framework_name = 'CaptionGeneratorFramework'

    def compile(self):
        # initialize model
        self.model = model_switcher(self.model_name)(
            self.input_shape,
            self.hidden_size,
            self.vocab_size,
            embedding_size=self.embedding_size,
            seed=self.random_seed)
        print(self.model)
        self.train_params = sum(p.numel() for p in self.model.parameters()
                                if p.requires_grad)
        print('Trainable Parameters:', self.train_params, '\n\n\n\n')
        self.initialize_optimizer()  # initialize optimizer

    def initialize_optimizer(self):
        self.optimizer = optimizer_switcher(self.optimizer_string)(
            self.model.parameters(), self.lr)

    def train(self,
              data_path,
              validation_path,
              ann_path,
              epochs,
              batch_size,
              early_stopping_freq=6,
              beam_size=1,
              validation_metric='CIDEr'):
        """
        Method for training the model.

        Parameters
        ----------
        data_path : Path or str.
            Path to trainingset file (*.csv).
        validation_path : Path or str.
            Path to validationset file (*.csv).
        ann_path : Path or str.
            Path to the validation annotation file (*.json)
        epochs : int.
            Max number of epochs to continue training the model for.
        batch_size : int.
            Mini batch size.
        early_stopping_freq : int.
            If no improvements over this number of epochs then stop training.
        beam_size : int.
            Beam size for validation. Default is 1.
        validation_metric : str.
            Which automatic text evaluation metric to use for validation.
            Metrics = {'CIDEr', 'METEOR', 'SPICE', 'ROUGE_L',
            'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4'}.
            Defualt value is 'CIDEr'.

        Returns
        -------
        Saves the model and checkpoints to its own folder. Then writes a log
        with all necessary information about the model and training.
        """
        data_path = Path(data_path)
        validation_path = Path(validation_path)
        ann_path = Path(ann_path)
        train_df = pd.read_csv(data_path)

        training_history = {
            'network': str(self.model),
            'trainable_parameters': str(self.train_params),
            'lr': str(self.lr),
            'optimizer': self.optimizer_string,
            'loss': self.loss_string,
            'model_name': self.model_name,
            'history': [],
            'voc_size': str(self.vocab_size),
            'voc_path': str(self.vocab_path),
            'feature_path': str(self.feature_path),
            'train_path': str(data_path),
            'epochs': str(epochs),
            'batch_size': str(batch_size),
            'training_time': str(0),
            'model_save_path': ''
        }

        steps_per_epoch = len(train_df) // batch_size

        train_generator = data_generator(train_df, batch_size,
                                         steps_per_epoch,
                                         self.wordtoix,
                                         self.encoded_features,
                                         seed=self.random_seed)

        start_time = time()

        date_time_obj = datetime.now()
        timestamp_str = date_time_obj.strftime("%d-%b-%Y_(%H:%M:%S)")
        directory = self.save_path.joinpath(self.model_name + '_'
                                            + timestamp_str)
        # check that directory is a Directory if not make it one
        if not directory.is_dir():
            directory.mkdir()

        best_val_score = -1  # all metric give positive scores
        best_path = None
        epochs_since_improvement = 0

        for e in range(1, epochs + 1):
            # early stopping
            if epochs_since_improvement == early_stopping_freq:
                print('Training TERMINATED!\nNo Improvements for '
                      + str(early_stopping_freq) + ' epochs.')
                break

            self.model.train()  # put model in train mode

            print('Epoch: #' + str(e))
            batch_history = []
            for s in range(1, steps_per_epoch + 1):
                print('Step: #' + str(s) + '/' + str(steps_per_epoch))
                # zero the gradient buffers
                self.optimizer.zero_grad()

                # get minibatch from data generator
                x, target, caption_lengths = next(train_generator)

                # get predictions from network
                output, decoding_lengths = self.model(x, caption_lengths)
                output = pack_padded_sequence(output,
                                              decoding_lengths,
                                              batch_first=True)[0]
                target = pack_padded_sequence(target,
                                              decoding_lengths,
                                              batch_first=True)[0]

                # get loss
                loss = self.criterion(output, target)
                loss_num = loss.item()
                batch_history.append(loss_num)
                print('loss', '(' + self.optimizer_string + '):', loss_num)
                # backpropagate
                loss.backward()
                # update weights
                self.optimizer.step()
            # add the mean loss of the epoch to the training history
            training_history['history'].append(np.mean(
                np.array(batch_history)))

            # validation here
            eval_path = directory.joinpath('captions_eval_' + str(e) + '.json')
            res_path = directory.joinpath('captions_result_' + str(e)
                                          + '.json')
            metric_score = self.evaluate(validation_path,
                                         ann_path,
                                         res_path,
                                         eval_path,
                                         beam_size=beam_size,
                                         metric=validation_metric)
            # save model checkpoint
            is_best = metric_score > best_val_score
            best_val_score = max(metric_score, best_val_score)
            tmp_model_path = save_checkpoint(directory,
                                             epoch=e,
                                             epochs_since_improvement=0,
                                             model=self.model,
                                             optimizer=self.optimizer,
                                             cider=metric_score,
                                             is_best=is_best)
            if tmp_model_path:
                best_path = tmp_model_path

            if is_best:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

        # end of training
        training_time = timedelta(seconds=int(time() - start_time))  # seconds
        d = datetime(1, 1, 1) + training_time
        training_time = "%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second)
        training_history['training_time'] = str(training_time)
        training_history['model_save_path'] = str(best_path)
        train_path = directory.joinpath(self.model_name + '_log.txt')
        # save model to file
        self.save_model(directory)
        # save log to file
        save_training_log(train_path, training_history)

    def predict(self, data_path, batch_size=1, beam_size=1):
        """
        Function to make self.model make predictions given some data.

        Parameters
        ----------
        data_path : Path or str.
            Path to csv file containing the test set.
        batch_size : int.
            The number of images to predict on simultaneously. Default 1.
        beam_size : int.
            Default is 1, which is the same as doing greedy inference.

        Returns
        -------
        predicted_captions : dict.
            Dictionary image_name as keys and predicted captions through
            beam search are the values.
        """
        data_path = Path(data_path)
        self.model.eval()  # put model in evaluation mode

        data_df = pd.read_csv(data_path)

        predicted_captions = []

        print_idx = 10
        num_images = len(data_df)
        if batch_size > num_images:
            batch_size = num_images
        steps = np.ceil(num_images / batch_size)
        prev_batch_idx = 0
        for i in range(steps):
            # create input batch
            end_batch_idx = min(prev_batch_idx + batch_size, num_images)
            image_ids = data_df.loc[prev_batch_idx: end_batch_idx,
                                    'image_id']
            image_names = data_df.loc[prev_batch_idx: end_batch_idx,
                                      'image_name']
            enc_images = []
            for image_id, image_name in zip(image_ids, image_names):
                # get encoded features for this batch
                pred_dict = {
                    "image_id": image_id
                }
                predicted_captions.append(pred_dict)
                enc_images.append(self.encoded_features[image_name])

            # get full sentence predictions from beam_search algorithm
            predictions = self.beam_search(enc_images, beam_size=beam_size)
            predictions = self.post_process_predictions(predictions)

            # put predictions in the right pred_dict
            counter = 0
            for idx in range(prev_batch_idx, end_batch_idx):
                predicted_captions[idx]["caption"] = predictions[counter]
                counter += 1

            # verbose
            index = i + 1
            if index % print_idx == 0:
                print('Batch step',  index)
            # update prev_batch_idx
            prev_batch_idx = end_batch_idx

        return predicted_captions

    @staticmethod
    def post_process_predictions(predictions):
        # helper function, consider moving to utils
        # remove startseq and endseq token from sequence
        processed = []
        for prediction in predictions:
            prediction = prediction.replace('startseq', '')
            prediction = prediction.replace('endseq', '')
            prediction = prediction.strip()
            processed.append(prediction)
        return processed

    def beam_search(self, batch, beam_size=1):
        # consider if it is possible to handle more than one sample at a time
        # for instance more images, and/or predict on the entire beam
        # initialization
        batch_size = len(batch)
        # init all image_seqs with sartseq and 0.0 prob
        in_token = [self.wordtoix['startseq']]
        captions = [[in_token, 0.0] for _ in range(batch_size)]

        # overhead
        batch_beam_size = [beam_size for _ in range(batch_size)]
        # initialize beams as containing 1 caption
        # need beams to keep track of original indices
        beams = [[[in_token, 0.0]] for _ in range(batch_size)]

        images = torch.tensor(batch)  # convert to tensor (bs, 64, 1536)
        predictions = defaultdict(str)  # key: batch index, val: caption

        while True:
            # size of tmp_captions is max b^2
            # find current batch_size
            batch_size_t = sum(bs > 0 for bs in batch_beam_size)

            sequences = [cap[0] for cap in captions]

            caption_lengths = np.array([len(s) for s in sequences])




            # TODO: remove the old under here when finished

            tmp_captions = []
            for caption in captions:
                # if this process proves to be too computationally heavy
                # then consider trade off with memory, by having extra
                # variable with both index rep and string rep.
                sequence = [self.wordtoix[w] for w in caption[0]
                            if w in self.wordtoix]
                sequence = torch.tensor(sequence)  # convert to tensor
                caption_lengths = np.array([sequence.size()[0]])
                # pad sequence
                sequence = pad_sequences([sequence], maxlen=self.max_length)

                # get predictions
                x = [image, sequence]
                y_predictions, decoding_lengths = \
                    self.model(x, caption_lengths, has_end_seq_token=False)
                # pack padded sequence
                y_predictions = pack_padded_sequence(y_predictions,
                                                     decoding_lengths,
                                                     batch_first=True)[0]
                # get the b most probable indices
                # first turn predictions into numpy array,
                # so that we can use np.argsort
                y_predictions = y_predictions.detach().numpy()[-1]
                words_predicted = np.argsort(y_predictions)[-beam_size:]
                for word in words_predicted:
                    new_partial_cap = deepcopy(caption[0])
                    # add the predicted word to the partial caption
                    new_partial_cap.append(self.ixtoword[word])
                    new_partial_cap_prob = caption[1] + y_predictions[word]
                    # add cap and prob to tmp list
                    tmp_captions.append([new_partial_cap,
                                         new_partial_cap_prob])
            captions = tmp_captions
            captions.sort(key=lambda l: l[1])
            captions = captions[-beam_size:]
        most_prob_cap = captions[-1][0]
        most_prob_cap = ' '.join(most_prob_cap)

        # should be removed when finished
        assert len(predictions) == batch_size, "The number of predictions " \
                                               "does not match the number " \
                                               "of images"
        return most_prob_cap.strip()

    def evaluate(self, data_path, ann_path, res_path, eval_path,
                 beam_size=1, metric='CIDEr'):
        print("Evaluating model ...")
        # get models predictions
        predictions = self.predict(data_path, beam_size=beam_size)
        print("Finished predicting")
        # save predictions to res_path which is .json
        with open(res_path, 'w') as res_file:
            json.dump(predictions, res_file)

        coco = COCO(str(ann_path))
        coco_res = coco.loadRes(str(res_path))
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.params['image_id'] = coco_res.getImgIds()
        coco_eval.evaluate()

        # save evaluations to eval_path which is .json
        with open(eval_path, 'w') as eval_file:
            json.dump(coco_eval.eval, eval_file)

        return coco_eval.eval[metric]

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.optimizer = checkpoint['optimizer']
        self.model.eval()
        print('Loaded checkpoint at:', path)
        print(self.model)

    def save_model(self, path):
        path = Path(path)
        assert path.is_dir()
        path = path.joinpath(self.model_name + '_model.pth')
        torch.save(self.model.state_dict(), path)

    def get_model(self):
        return self.model

    def set_model(self, model):
        # expects a pytorch model
        self.model = model

    def get_model_name(self):
        return self.model_name

    def set_model_name(self, string):
        assert isinstance(string, str)
        self.model_name = string

    def get_max_length(self):
        return self.max_length

    def set_max_length(self, length):
        assert isinstance(length, int)
        self.max_length = length
