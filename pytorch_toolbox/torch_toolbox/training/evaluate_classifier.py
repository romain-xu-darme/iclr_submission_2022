#!/usr/bin/env python3
from torch_toolbox.training.training import (
    evaluation_add_parser_options,
    evaluation_init,
    evaluation_loop,
)
from torch_toolbox.training.loss import FocalLoss
import argparse


def _main():
    # Parse command line
    ap = argparse.ArgumentParser(
        description='Evaluate model on dataset.')
    evaluation_add_parser_options(ap)
    ap.add_argument('--focal-loss-gamma', required=False, type=float,
                    metavar='<gamma>',
                    default=0.0,
                    help='Gamma parameter of the Focal loss -1*(1-pt)^gamma * log(pt)')

    # Parse arguments
    args = ap.parse_args()

    # Init evaluation and parse common options
    model, dataloaders = evaluation_init(args)

    # Launch evaluation
    evaluation_loop(
        model=model,
        dataloaders=dataloaders,
        loss_fn=FocalLoss(gamma=args.focal_loss_gamma),
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    _main()
