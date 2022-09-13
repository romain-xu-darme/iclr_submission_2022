#!/usr/bin/env python3
import math
import torch
import torch_toolbox
import numpy as np
from torch_toolbox.datasets.dataset import dataloader_add_parser_options, dataloaders_from_parser
from progress.bar import Bar
from typing import Optional, Iterable
from scipy.stats import logistic
from argparse import RawTextHelpFormatter
import argparse


def _main():
    ap = argparse.ArgumentParser(
        description='Calibrate a Particul model.',
        formatter_class=RawTextHelpFormatter)
    ap.add_argument('-i', '--input', required=True, type=str,
                    metavar='<path_to_file>',
                    help='Path to input model.')
    ap.add_argument("-o", "--output", required=True, type=str,
                    metavar='<path_to_file>',
                    help="Path to output model")
    ap.add_argument("--npy", required=False, type=str,
                    metavar='<path_to_file>',
                    help="Path to NPY output file")
    ap.add_argument('--device', type=str, required=False,
                    metavar='<device_id>',
                    default='cpu',
                    help='Target device for execution.')
    ap.add_argument("--plot", action='store_true',
                    help="Plot distributions")
    ap.add_argument('-v', "--verbose", action='store_true',
                    help="Verbose mode")

    # Add dataset options
    dataloader_add_parser_options(ap)
    args = ap.parse_args()

    # Load datasets
    trainset, testset, _ = dataloaders_from_parser(args)

    # Load model, set evaluation mode and disable normalization
    trained_model = torch_toolbox.load(args.input, map_location=args.device)

    # By default, assume trained model is a Particul module
    model = trained_model
    if trained_model.__class__.__name__ in ['ParticulCNet', 'ParticulPNet']:
        model = trained_model.extractor

    # Save configuration
    e_feat = model.enable_features
    e_norm = model.enable_normalization
    e_conf = model.enable_confidence
    model.eval()
    model.enable_normalization = False
    model.enable_features = False
    model.enable_confidence = False

    npatterns = model.npatterns

    if args.plot:
        import matplotlib.pyplot as plt
        plots_per_row = 3
        nrows = int(math.ceil(npatterns / plots_per_row))
        fig, axs = plt.subplots(nrows=2 * nrows, ncols=plots_per_row, sharex='col')

    def get_max_scores(dataset: Iterable, name: Optional[str] = 'dataset') -> np.array:
        """ Compute max correlation scores for all detectors across a dataset

        :param dataset: Dataset
        :param name: Dataset name
        :return: numpy array containing max correlation scores for all dataset inputs
        """
        res = [[] for _ in range(npatterns)]
        bar = Bar(f'Computing scores on {name}',
                  max=len(dataset), suffix='%(percent).1f%%') if args.verbose else None
        for imgs, _ in dataset:
            # Move to device
            if isinstance(imgs, list):
                imgs = torch.stack(imgs)
            imgs = imgs.to(args.device, non_blocking=True)
            # Compute correlation scores
            cscores = model(imgs).detach().cpu().numpy().copy()
            # Recover maximum value of pattern correlations
            vmax = np.max(cscores, axis=(2, 3))
            for index in range(npatterns):
                res[index] += list(vmax[..., index])
            if args.verbose:
                bar.next()
        if args.verbose:
            bar.finish()
        return np.array(res)

    m_val = get_max_scores(trainset, 'trainset')
    t_val = get_max_scores(testset, 'testset') if args.plot else None

    if args.npy:
        np.save(f'train_{args.npy}', np.array(m_val), allow_pickle=True)
        if t_val is not None:
            np.save(f'test_{args.npy}', np.array(t_val), allow_pickle=True)


    for i in range(npatterns):
        # Get mean and standard deviation of distribution
        # of max correlation scores for a given pattern detector
        mu, sigma = logistic.fit(m_val[i])
        if args.plot:
            r, c = int(i / plots_per_row), i % plots_per_row
            X = np.arange(mu - 20, mu + 20, 1)
            # Plot Probability Distribution Function (PDF) computed on trainset
            axs[2 * r, c].plot(X, [logistic.pdf(x, mu, sigma) for x in X], color='tab:green', label='Trainset')
            # Plot test set histogram
            axs[2 * r, c].hist([t_val[i]], bins=100, density=True, color='tab:blue', alpha=0.5, label='Testset')
            axs[2 * r, c].legend()
            # Plot Cumulative Distribution Function (CDF) computed on trainset
            axs[2 * r + 1, c].plot(X, [logistic.cdf(x, mu, sigma) for x in X], color='tab:purple',
                                   label='Confidence measure')
            axs[2 * r + 1, c].legend()
            axs[2 * r, c].set_xlabel = f'Pattern {i} max correlation score'
            axs[2 * r, c].set_ylabel = f'Distribution'
        if args.verbose:
            print(f"Distribution for pattern {i}: Mean={mu}, std={sigma}")

        # Update model distribution parameters
        model.detectors[i].calibrate(mean=mu, std=sigma)

    # Restore configuration
    model.enable_features = e_feat
    model.enable_normalization = e_norm
    model.enable_confidence = e_conf

    # Save model
    torch_toolbox.save(trained_model, args.output)

    if args.plot:
        plt.show()


if __name__ == '__main__':
    _main()
