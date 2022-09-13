#!/usr/bin/env python3
import torch_toolbox
import torch.nn as nn
from torch_toolbox.training.training import (
    training_add_parser_options,
    training_init,
    training_loop,
)
from particul.od2.model import ClassifierWithParticulOD2, ParticulOD2
from particul.od2.loss import ParticulOD2Loss
import argparse
from argparse import RawTextHelpFormatter
from typing import List


def _main():
    # Parse command line
    ap = argparse.ArgumentParser(
        description='Train a Particul-based Out-of-Distribution detector.',
        formatter_class=RawTextHelpFormatter)
    ap.add_argument('--od2-classifier', required=True, type=str,
                    metavar='<path_to_file>',
                    help='Path to classifier.')
    ParticulOD2Loss.add_parser_options(ap)
    training_add_parser_options(ap)

    # Parse arguments
    args = ap.parse_args()

    # Init training and parse common options
    detector, dataloaders, opt, scheduler, init_epoch, best_loss = training_init(args)

    # Load classifier (not attached to optimizer)
    classifier = torch_toolbox.load(args.od2_classifier, map_location=args.device)

    # Combine target classifier with ParticulOD2 module
    model = ClassifierWithParticulOD2(classifier, detector)
    model.mode = 'logits+amaps'

    # Build loss and decision callback
    npatterns = model.detector.npatterns
    loss_fn = ParticulOD2Loss.build_from_parser(npatterns, args).to(args.device)

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
                torch_toolbox.save(model.detector, path)
            return self.best_loss

    class LoadingCallback:
        def __init__(self, classifier: nn.Module) -> None:
            """ Build loading callback

            :param classifier: Target classifier
            """
            self.classifier = classifier

        def __call__(self, path: str, device: str) -> nn.Module:
            """ Load best model

            :param path: Path to output file
            :param device: Target device
            """
            return ClassifierWithParticulOD2(self.classifier,
                                             ParticulOD2.load(path)).to(args.device)

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
        loading_callback=LoadingCallback(classifier)
    )


if __name__ == '__main__':
    _main()
