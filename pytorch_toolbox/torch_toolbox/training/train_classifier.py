#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch_toolbox.training.training import (
    training_add_parser_options,
    training_init,
    training_loop,
)
from torch_toolbox.training.loss import FocalLoss
import argparse
from argparse import RawTextHelpFormatter
from typing import List


class DecisionCallback:
    def __init__(self, policy: str, best_metrics: List[float], verbose: bool) -> None:
        """ Build decision callback

        :param policy: Save best model w.r.t. to either loss or accuracy
        :param best_metrics: Current best loss and accuracy
        :param verbose: Enable verbose mode
        """
        self.policy = policy
        self.best_metrics = best_metrics if best_metrics is not None else [50, 0]
        self.verbose = verbose

    def __call__(self, model: nn.Module, path: str, metrics: List[float]) -> List[float]:
        """ Save model if the epoch metrics are better than current values

        :param model: Current model
        :param path: Path to output file
        :param metrics: Epoch metrics
        :return: Current best metrics
        """
        if (self.policy == 'loss' and metrics[0] < self.best_metrics[0]) or \
                (self.policy == 'accuracy' and metrics[1] > self.best_metrics[1]):
            if self.verbose:
                print(f'Saving best model into {path}')
            self.best_metrics = metrics
            torch.save(model, path)
        return self.best_metrics


def _main():
    # Parse command line
    ap = argparse.ArgumentParser(
        description='Train a classifier model.',
        formatter_class=RawTextHelpFormatter)
    # Add common training options
    training_add_parser_options(ap)
    ap.add_argument('--save-best', required=False, type=str,
                    metavar='<metric>',
                    default='accuracy',
                    choices=['loss', 'accuracy'],
                    help='Set the metric defining the best model (default: accuracy)')
    ap.add_argument('--focal-loss-gamma', required=False, type=float,
                    metavar='<gamma>',
                    default=0.0,
                    help='Gamma parameter of the Focal loss -1*(1-pt)^gamma * log(pt)')

    # Parse arguments
    args = ap.parse_args()

    # Init training and parse common options
    model, dataloaders, opt, scheduler, init_epoch, metrics = \
        training_init(args)

    # Launch training
    training_loop(
        model=model,
        dataloaders=dataloaders,
        init_epoch=init_epoch,
        num_epochs=args.epochs,
        opt=opt,
        scheduler=scheduler,
        loss_fn=FocalLoss(gamma=args.focal_loss_gamma),
        output=args.output,
        decision_callback=DecisionCallback(args.save_best, metrics, args.verbose),
        checkpoint_file=args.checkpoint_to,
        checkpoint_freq=args.checkpoint_every,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    _main()
