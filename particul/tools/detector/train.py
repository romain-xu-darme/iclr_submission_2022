#!/usr/bin/env python3
import torch_toolbox
import torch.nn as nn
from torch_toolbox.training.training import (
    training_add_parser_options,
    training_init,
    training_loop,
)
from particul.detector.loss import ParticulLoss
import argparse
from argparse import RawTextHelpFormatter
from typing import List

# Parse command line
ap = argparse.ArgumentParser(
    description='Train a Particul model.', formatter_class=RawTextHelpFormatter)
training_add_parser_options(ap)
ParticulLoss.add_parser_options(ap)
ap.add_argument('--unfreeze-backbone', required=False, action='store_true',
                help='Unfreeze backbone layers.')

# Parse arguments
args = ap.parse_args()

# Init training and parse common options
model, dataloaders, opt, scheduler, init_epoch, best_loss = training_init(args)

# Unfreeze backbone if necessary
if args.unfreeze_backbone:
    model.train_backbone(True)

# Build loss and decision callback
loss_fn = ParticulLoss.build_from_parser(model.npatterns, args).to(args.device)


class DecisionCallback:
    def __init__(self, best_loss: float, verbose: bool) -> None:
        """ Build decision callback

        :param best_loss: Current best loss
        :param verbose: Enable verbose mode
        """
        self.best_loss = best_loss if best_loss is not None else 50
        self.verbose = verbose

    def __call__(self, model: nn.Module, path: str, metrics: List[float]) -> float:
        """ Save model if the epoch loss is better than current value

        :param model: Current model
        :param path: Path to output file
        :param metrics: Epoch metrics
        :return: Current best loss
        """
        if metrics[0] < self.best_loss:
            if self.verbose:
                print(f'Saving best model into {path}')
            self.best_loss = metrics[0]
            torch_toolbox.save(model, path)
        return self.best_loss


# Launch training
training_loop(
    model=model,
    dataloaders=dataloaders,
    init_epoch=init_epoch,
    num_epochs=args.epochs,
    opt=opt,
    scheduler=scheduler,
    loss_fn=loss_fn,
    output=args.output,
    decision_callback=DecisionCallback(best_loss, args.verbose),
    checkpoint_file=args.checkpoint_to,
    checkpoint_freq=args.checkpoint_every,
    device=args.device,
    verbose=args.verbose,
)
