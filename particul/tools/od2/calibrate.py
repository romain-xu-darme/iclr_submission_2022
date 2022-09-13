#!/usr/bin/env python3
import torch
import torch_toolbox
import numpy as np
from particul.od2.model import ClassifierWithParticulOD2
from torch_toolbox.datasets.dataset import (
    dataloader_add_parser_options,
    dataloaders_from_parser,
)
from progress.bar import Bar
from typing import Optional, Sized, List
from scipy.stats import logistic
from argparse import RawTextHelpFormatter
import argparse


def _main():
    ap = argparse.ArgumentParser(
        description='Calibrate a Particul-based Out-of-Distribution detector.',
        formatter_class=RawTextHelpFormatter)
    ClassifierWithParticulOD2.add_parser_options(ap)
    ap.add_argument("--use-labels", action='store_true',
                    help="Use dataset labels")
    ap.add_argument("-o", "--output", required=True, type=str,
                    metavar='<path_to_file>',
                    help="Path to output model")
    ap.add_argument('--device', type=str, required=False,
                    metavar='<device_id>',
                    default='cpu',
                    help='Target device for execution.')
    ap.add_argument("--plot", type=int, required=False, nargs=2,
                    metavar=('<class_index>', '<class_num>'),
                    help="Plot distributions for a given set of classes")
    ap.add_argument('-v', "--verbose", action='store_true',
                    help="Verbose mode")

    # Add dataset options
    dataloader_add_parser_options(ap)
    args = ap.parse_args()

    # Load datasets
    trainset, testset, _ = dataloaders_from_parser(args)

    model = ClassifierWithParticulOD2.build_from_parser(args)
    # Set evaluation mode and disable normalization
    model.eval()
    model.detector.enable_normalization = False
    npatterns = model.detector.npatterns
    nclasses = model.detector.nclasses
    model.mode = "logits+amaps"

    if args.plot:
        # In plot
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=args.plot[1], ncols=npatterns, sharex='col')

    def get_max_scores(dataset: Sized, name: Optional[str] = 'dataset') -> List:
        """ Compute max correlation scores for all detectors across a dataset

        :param dataset: Dataset
        :param name: Dataset name
        :return: List containing max correlation scores. Shape nclasses x npatterns x ?
        """
        res = [[[] for _ in range(npatterns)] for _ in range(nclasses)]
        bar = Bar(f'Computing scores on {name}',
                  max=len(dataset), suffix='%(percent).1f%%') if args.verbose else None

        for imgs, labels in dataset:
            # Move to device
            if isinstance(imgs, list):
                imgs = torch.stack(imgs)
            imgs = imgs.to(args.device, non_blocking=True)
            # Compute prediction and correlation scores. Shape (N x C), (N x C x H x W x P)
            _, amaps = model(imgs)
            amaps = amaps.detach().cpu().numpy()  # Shape N x C x P x H x W
            # Recover maximum value of pattern correlations. Shape N x C x P
            vmax = np.max(amaps, axis=(3, 4))
            if args.use_labels:
                labels = labels.detach().numpy()
                if labels.ndim > 1:
                    labels = labels[..., 0]  # Shape N
                for v, label in zip(vmax, labels):
                    for p in range(npatterns):
                        res[label][p].append(v[label, p])
            else:
                raise ValueError('Not implemented yet')

            if args.verbose:
                bar.next()
        if args.verbose:
            bar.finish()
        return res

    m_val = get_max_scores(trainset, 'trainset')
    if args.plot:
        t_val = get_max_scores(testset, 'testset')

    for c in range(nclasses):
        for p in range(npatterns):
            # Get mean and standard deviation of distribution
            # of max correlation scores for a given pattern detector
            mu, sigma = logistic.fit(m_val[c][p])
            if args.plot and c in range(args.plot[0], args.plot[0] + args.plot[1]):
                r = c - args.plot[0]
                x_range = np.arange(mu - 5 * sigma, mu + 5 * sigma, sigma / 20)
                # Plot Probability Distribution Function (PDF) computed on trainset
                axs[r, p].plot(x_range, [logistic.pdf(x, mu, sigma) for x in x_range],
                               color='tab:green', label='Trainset')
                axs[r, p].hist([m_val[c][p]], bins=100, density=True,
                               color='tab:orange', alpha=0.5, label='Trainset')
                # Plot test set histogram
                axs[r, p].hist([t_val[c][p]], bins=100, density=True,
                               color='tab:blue', alpha=0.5, label='Testset')
                axs[r, p].legend()
                axs[r, p].set_xlabel = f'Class {c}/Pattern {p} max correlation score'
                axs[r, p].set_ylabel = f'Distribution'
            if args.verbose:
                print(f"Distribution for class {c} pattern {p}: Mean={mu}, std={sigma}")

            # Update model distribution parameters
            model.detector.particuls[c].detectors[p].calibrate(mean=mu, std=sigma)

    # Restore configuration
    model.detector.enable_normalization = True

    # Save model
    torch_toolbox.save(model.detector, args.output)

    if args.plot:
        plt.show()


if __name__ == '__main__':
    _main()
