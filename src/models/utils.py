import torch
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parents[2]


def save_checkpoint(directory,
                    epoch,
                    epochs_since_improvement,
                    model,
                    optimizer,
                    cider,
                    is_best):
    directory = Path(directory)
    # check that directory is a Directory if not make it one
    if not directory.is_dir():
        directory.mkdir()

    # remove worse checkpoints
    for file in directory.glob('checkpoint_*'):
        file.unlink()

    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'cider': cider,
        'model': model,
        'optimizer': optimizer,
    }

    filename = directory.joinpath('checkpoint_' + str(epoch) + '.pth.tar')
    torch.save(state, filename)
    # If this checkpoint is the best so far,
    # store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        # remove last best checkpoint
        for file in directory.glob('BEST_*'):
            file.unlink()

        filename = directory.joinpath('BEST_checkpoint_' + str(epoch)
                                      + '.pth.tar')
        torch.save(state, filename)
