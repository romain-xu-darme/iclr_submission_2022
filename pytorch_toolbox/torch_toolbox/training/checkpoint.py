import torch
import torch.nn as nn
from torch.optim import Optimizer
from argparse import ArgumentParser
from typing import Any, Tuple, Optional
import os


def save_checkpoint(
        filename: str,
        model: nn.Module,
        metrics: Any,
        epoch: int,
        opt: Optimizer,
        freq: int,
        verbose: Optional[bool] = False,
) -> None:
    """ Save current training state

    :param filename: Path to file
    :param model: Current model
    :param metrics: Relevant metrics
    :param epoch: Index of last epoch
    :param opt: Optimizer
    :param freq: Number of epochs between checkpoints
    :param verbose: Verbose mode
    """
    if (freq > 0) and (epoch % freq == 0) and (filename is not None):
        if verbose:
            print(f'Epoch {epoch}: Saving checkpoint to {filename}')
        state = {
            'metrics': metrics,
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer': opt.state_dict()
        }
        torch.save(state, filename)


def load_checkpoint(
        filename: str,
        model: nn.Module,
        opt: Optimizer,
        device: str,
        verbose: Optional[bool] = False,
) -> Tuple[Any, int]:
    """ Load training state

    :param filename: Path to file
    :param model: Model
    :param device: Target device
    :param opt: Optimizer
    :param verbose: Verbose mode
    :return: Epoch to restart from, metrics
    """
    if not os.path.isfile(filename):
        raise ValueError(f'{filename} not found')

    # Load checkpoint
    checkpoint = torch.load(filename, map_location=device)

    # Set model and optimizer state
    model.load_state_dict(checkpoint['model_state'])
    opt.load_state_dict(checkpoint['optimizer'])

    # Best loss, best accuracy and initial epoch
    metrics = checkpoint['metrics']
    init_epoch = checkpoint['epoch']

    if verbose:
        print(f'Checkpoint {filename} loaded.\n \
                \t > Epoch: {init_epoch}\n \
                \t > Metrics: {metrics}')
    return init_epoch, metrics


def checkpoint_add_parser_options(parser: ArgumentParser) -> None:
    """ Add checkpoint options to ArgumentParser.

    :param parser: Existing parser
    """
    parser.add_argument('--checkpoint-to', required=False, type=str,
                        metavar='<path_to_file>',
                        default='checkpoint.pth.tar',
                        help='Path to output checkpoint file (default: checkpoint.pth.tar)')
    parser.add_argument('--checkpoint-from', required=False, type=str,
                        metavar='<path_to_file>',
                        help='Path to input checkpoint file (resume computation)')
    parser.add_argument('--checkpoint-every', required=False, type=int,
                        metavar='<num_epochs>',
                        default=0,
                        help='Save checkpoint every n epochs (default: 0 -> disabled)')
