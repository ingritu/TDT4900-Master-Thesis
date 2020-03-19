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
from time import time
import json

from src.data.data_generator import data_generator
from src.data.utils import load_vocabulary
from src.features.resnet_features import load_visual_features
from src.models.torch_generators import model_switcher
from src.models.beam import Beam
from src.models.utils import save_checkpoint
from src.models.utils import save_training_log

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

ROOT_PATH = Path(__file__).absolute().parents[2]


def loss_switcher(loss_string):
    loss_string = loss_string.lower()
    switcher = {
        'cross_entropy': nn.CrossEntropyLoss,
        'mse': nn.MSELoss
    }

    return switcher.get(loss_string, nn.CrossEntropyLoss)


def optimizer_switcher(optimizer_string):
    optimizer_string = optimizer_string.lower()
    switcher = {
        'adam': optim.Adam,
        'sgd': optim.SGD
    }
    return switcher.get(optimizer_string, optim.SGD)


class Generator:

    def __init__(self,
                 model_name,
                 voc_path,
                 feature_path):
        """
        Generator framework similar to how keras works.

        Parameters
        ----------
        model_name : str.
        voc_path : Path or str.
        feature_path : Path or str.
        """
        self.embedding_size = 0
        self.hidden_size = 0

        self.save_path = None

        self.vocab_path = Path(voc_path)
        self.wordtoix, self.ixtoword, self.max_length = \
            load_vocabulary(self.vocab_path)
        self.vocab_size = len(self.wordtoix)
        self.feature_path = Path(feature_path)
        self.encoded_features = load_visual_features(self.feature_path)
        self.input_shape = list(self.encoded_features.values())[0].shape

        # initialize model as None
        self.model = None
        self.train_params = 0

        self.model_name = model_name

        # initialize loss function
        self.loss_string = ""
        self.criterion = None

        # set up optimizer
        self.optimizer_string = ""
        self.optimizer = None  # not initialized
        self.lr = 0

        # misc
        self.framework_name = 'CaptionGeneratorFramework'
        # gpu
        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available() else "cpu")
        print('Device:', self.device)

    def compile(self,
                save_path,
                embedding_size=512,
                hidden_size=512,
                loss_function='cross_entropy',
                optimizer='adam',
                lr=0.0005):
        """
        Bulids the model.

        Parameters
        ----------
        save_path : Path or str.
        embedding_size : int.
        hidden_size : int.
        loss_function : str.
        optimizer : str.
        lr : float.
        """
        # set values
        self.save_path = Path(save_path)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.loss_string = loss_function
        self.criterion = loss_switcher(self.loss_string)()
        self.optimizer_string = optimizer
        self.lr = lr

        # initialize model
        self.model = model_switcher(self.model_name)(self.input_shape,
                                                     self.max_length,
                                                     self.hidden_size,
                                                     self.vocab_size,
                                                     self.device,
                                                     self.embedding_size)

        print(self.model)
        self.train_params += sum(p.numel() for p in self.model.parameters()
                                 if p.requires_grad)
        print('Trainable Parameters:', self.train_params, '\n\n\n\n')

        # check for multiple GPUs
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.initialize_optimizer()

    def initialize_optimizer(self):
        """
        Initializes the optimizer.
        After this is called, optimizer will no longer be None.
        """
        self.optimizer = optimizer_switcher(self.optimizer_string)(
            self.model.parameters(), self.lr)

    def train(self,
              data_path,
              validation_path,
              ann_path,
              epochs,
              batch_size,
              early_stopping_freq=6,
              val_batch_size=1,
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
        val_batch_size : int.
            The number of images to do inference on simultaneously.
            Default is 1.
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
            'model': str(self.model),
            'trainable_parameters': str(self.train_params),
            'lr': str(self.lr),
            'optimizer': self.optimizer_string,
            'loss': self.loss_string,
            'model_name': self.model_name,
            'history': [],
            'val_history': [],
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
                                         self.encoded_features)

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
                # get minibatch from data generator
                x, caption_lengths = next(train_generator)

                loss_num = self.train_on_batch(x, caption_lengths)
                batch_history.append(loss_num)
                print('Step: #' + str(s) + '/' + str(steps_per_epoch) + '\t' +
                      'loss (' + self.optimizer_string + '):', loss_num)

            # add the mean loss of the epoch to the training history
            training_history['history'].append(np.mean(
                np.array(batch_history)))

            # validation here
            # consider just overwriting the same file
            eval_path = directory.joinpath('captions_eval_' + str(e) + '.json')
            res_path = directory.joinpath('captions_result_' + str(e)
                                          + '.json')
            val_start = time()
            metric_score = self.evaluate(validation_path,
                                         ann_path,
                                         res_path,
                                         eval_path,
                                         batch_size=val_batch_size,
                                         beam_size=beam_size,
                                         metric=validation_metric)
            val_time = time() - val_start
            training_history['val_history'].append(metric_score)
            print("Validation took", round(val_time, 2), 'seconds')
            # save model checkpoint
            is_best = metric_score > best_val_score
            best_val_score = max(metric_score, best_val_score)
            tmp_model_path = save_checkpoint(
                directory,
                epoch=e,
                epochs_since_improvement=epochs_since_improvement,
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

    def train_on_batch(self, x, caption_lengths):
        """
        Do one epoch of training on batch x.

        Parameters
        ----------
        x : list.
            batch of images and captions.

        caption_lengths :

        Returns
        -------
        loss_num : float
        """
        predictions, target, decoding_lengths = self.model(x, caption_lengths)

        # loop finished
        # pack padded sequences
        output = pack_padded_sequence(predictions,
                                      decoding_lengths,
                                      batch_first=True)[0]
        target = pack_padded_sequence(target,
                                      decoding_lengths,
                                      batch_first=True)[0]

        # get loss
        loss = self.criterion(output, target)
        loss_num = loss.item()

        # backpropagate
        # zero the gradient buffers
        self.optimizer.zero_grad()
        loss.backward()
        # update weights
        self.optimizer.step()

        return loss_num

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
        predicted_captions : list.
            Dictionary image_name as keys and predicted captions through
            beam search are the values.
        """
        data_path = Path(data_path)
        self.model.eval()  # put model in evaluation mode

        data_df = pd.read_csv(data_path)
        data_df = data_df.reset_index(drop=True)

        predicted_captions = []

        num_images = len(data_df)
        if batch_size > num_images:
            batch_size = num_images
        steps = int(np.ceil(num_images / batch_size))
        prev_batch_idx = 0
        for i in range(steps):
            # create input batch
            end_batch_idx = min(prev_batch_idx + batch_size, num_images)
            batch_df = data_df.iloc[prev_batch_idx: end_batch_idx, :]
            image_ids = batch_df.loc[:, 'image_id'].to_numpy()
            image_names = batch_df.loc[:, 'image_name'].to_numpy()

            enc_images = []
            for image_id, image_name in zip(image_ids, image_names):
                # get encoded features for this batch
                pred_dict = {
                    "image_id": int(image_id),
                    "caption": ""
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
            print('Batch step', i + 1)
            # update prev_batch_idx
            prev_batch_idx = end_batch_idx

        return predicted_captions

    @staticmethod
    def post_process_predictions(predictions):
        # helper function, consider moving to utils
        # remove startseq and endseq token from sequence
        processed = []
        for prediction in predictions.values():
            prediction = prediction.replace('startseq', '')
            prediction = prediction.replace('endseq', '')
            prediction = prediction.strip()
            processed.append(prediction)
        return processed

    def beam_search(self, batch, beam_size=1):
        """
        Preform the beam search algorithm on a batch of images.

        Parameters
        ----------
        batch : list.
            List of encoded images.
        beam_size : int.
            Size of beam. Default is 1.

        Returns
        -------
        predictions : dict.
            key: image index, value: predicted caption.
        """
        # consider if it is possible to handle more than one sample at a time
        # for instance more images, and/or predict on the entire beam
        # initialization
        batch_size = len(batch)
        batch = torch.tensor(batch).to(self.device)

        global_images, encoded_images = self.model.encoder(batch)

        h_t, c_t = self.model.decoder.initialize_variables(
            batch_size * beam_size)

        # initialize beams as containing 1 caption
        # need beams to keep track of original indices
        beams = [Beam([g_image, enc_image],
                      states=[h_t[i*beam_size: (i+1)*beam_size],
                              c_t[i*beam_size: (i+1)*beam_size]],
                      beam_size=beam_size,
                      input_token=[self.wordtoix['startseq']],
                      eos=self.wordtoix['endseq'],
                      max_len=self.max_length,
                      vocab_size=self.vocab_size,
                      device=self.device,
                      beam_id=i)
                 for i, (g_image, enc_image) in
                 enumerate(zip(global_images, encoded_images))]

        working_beams_idx = set([i for i in range(batch_size)])

        predictions = defaultdict(str)  # key: batch index, val: caption

        while True:
            # find current batch_size aka number of beams
            # counts unfinished beams
            batch_size_t = sum(b.num_unfinished > 0 for b in beams)
            if batch_size_t == 0:
                # all beams are done
                break

            # get sequences
            sequences = torch.cat([b.get_sequences() for b in beams], dim=0)
            # get images
            global_images = torch.cat([b.get_global_image() for b in beams],
                                      dim=0)
            encoded_images = torch.cat([b.get_encoded_image() for b in beams],
                                       dim=0)
            images = [global_images, encoded_images]
            # get states
            h_t = torch.cat([b.get_hidden_states() for b in beams], dim=0)
            c_t = torch.cat([b.get_cell_states() for b in beams], dim=0)

            # get predictions
            x = [images, sequences]
            y_predictions, h_t, c_t = self.model.decoder(x, (h_t, c_t))
            # y_predictions (M*N, voc_size)
            y_predictions = torch.log_softmax(y_predictions, dim=1)
            # higher log_prob --> higher pob

            remove_idx = set()
            for i in range(batch_size):
                start_idx = i * beam_size
                if i in working_beams_idx:
                    # feed right predictions to right beams
                    b = beams[i]
                    end_idx = start_idx + b.num_unfinished
                    preds = y_predictions[start_idx: end_idx]
                    # update beam with predictions
                    b.update(preds,
                             h_t[start_idx: end_idx],
                             c_t[start_idx: end_idx])

                    if b.has_best_sequence():
                        # add to finished predictions
                        predictions[i] = \
                            ' '.join([self.ixtoword[w]
                                      for w in b.get_best_sequence()])
                        remove_idx.add(i)
            # remove idx of finished beams
            working_beams_idx = set(idx for idx in working_beams_idx
                                    if idx not in remove_idx)

        # should be removed when finished
        assert len(predictions) == batch_size, \
            "The number of predictions does not match the number of images"
        return predictions

    def evaluate(self, data_path, ann_path, res_path, eval_path, batch_size=1,
                 beam_size=1, metric='CIDEr'):
        """
        Function to evaluate model on data from data_path
        using evaluation metric metric.

        Parameters
        ----------
        data_path : Path or str.
            Validation or test set.
        ann_path : Path or str.
            Location of annotation file corresponding to val or test set.
        res_path : Path or str.
            File that the results will be saved in.
        eval_path : Path or str.
            File where all metric scores will be saved in.
        batch_size : int.
            The number of images to perform inference on simultaneously.
        beam_size : int.
            Size of beam.
        metric : str.
            Automatic evaluation metric. Compatible metrics are
            {Bleu_1, Bleu2, Bleu_3, Bleu_4, METEOR, ROUGE_L, CIDEr, SPICE}.

        Returns
        -------
        Automatic evaluation score.
        """
        data_path = Path(data_path)
        ann_path = Path(ann_path)
        res_path = Path(res_path)
        eval_path = Path(eval_path)
        print("Evaluating model ...")
        # get models predictions
        predictions = self.predict(data_path,
                                   batch_size=batch_size,
                                   beam_size=beam_size)
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
        """
        Load model from checkpoint.

        Parameters
        ----------
        path : Path or str.
        """
        path = Path(path)
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.optimizer = checkpoint['optimizer']
        self.model.eval()
        print('Loaded checkpoint at:', path)
        print(self.model)

    def save_model(self, path):
        """
        Save model to path.

        Parameters
        ----------
        path : Path or str.
        """
        path = Path(path)
        assert path.is_dir()
        path = path.joinpath(self.model_name + 'model.pth')
        torch.save(self.model.state_dict(), path)
