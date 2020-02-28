import torch
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parents[2]


def save_checkpoint(directory,
                    epoch,
                    epochs_since_improvement,
                    encoder,
                    decoder,
                    enc_optimizer,
                    dec_optimizer,
                    cider,
                    is_best):
    directory = Path(directory)

    # remove worse checkpoints
    for file in directory.glob('checkpoint_*'):
        file.unlink()

    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'cider': cider,
        'encoder': encoder,
        'decoder': decoder,
        'enc_optimizer': enc_optimizer,
        'dec_optimizer': dec_optimizer
    }

    filename = directory.joinpath('checkpoint_' + str(epoch) + '.pth.tar')
    torch.save(state, filename)
    # If this checkpoint is the best so far,
    # store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        # remove last best checkpoint
        for file in directory.glob('BEST_*'):
            file.unlink()

        filename = directory.joinpath('BEST_checkpoint.pth.tar')
        torch.save(state, filename)
        return filename
    return None


def save_training_log(path, training_history):
    # I do not care that the function is static
    with open(path, 'w') as train_log:
        train_log.write('################# '
                        'LOG FILE '
                        '#################\n\n')
        train_log.write('DATA and FEATURES\n')
        train_log.write('Training data path: ' +
                        training_history['train_path'] + '\n')
        train_log.write('Feature path: ' + training_history['feature_path']
                        + '\n')
        train_log.write('Vocabulary path: ' + training_history['voc_path']
                        + '\n')
        train_log.write('Vocabulary size: ' + training_history['voc_size']
                        + '\n\n')

        train_log.write('## CONFIGS / HYPERPARAMETERS ##\n')
        train_log.write('Optimizer: ' + training_history['optimizer'] +
                        '\n')
        train_log.write('Learning rate: ' + training_history['lr']
                        + '\n\n')

        train_log.write('####### '
                        'MODEL '
                        '#######\n')
        train_log.write('Model name: ' + training_history['model_name']
                        + '\n\n')
        train_log.write(training_history['encoder'] + '\n')
        train_log.write(training_history['decoder'] + '\n')
        train_log.write('Trainable parameters: ' +
                        training_history['trainable_parameters'] + '\n')
        train_log.write('Model save path: ' +
                        training_history['model_save_path'] + '\n\n')

        train_log.write('## Training Configs ##\n')
        train_log.write('Epochs: ' + training_history['epochs'] + '\n')
        train_log.write('Batch size: ' + training_history['batch_size']
                        + '\n')
        train_log.write('Training time: ' +
                        training_history['training_time'] + '\n')
        train_log.write('Loss function: ' + training_history['loss']
                        + '\n\n')
        train_log.write('## Train log!\n')
        # Lastly write the training log
        for loss in training_history['history']:
            # TODO: add val score.... zip?
            train_log.write(str(round(loss, 5)) + '\n')
