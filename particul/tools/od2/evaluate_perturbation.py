#!/usr/bin/env python3
import torch_toolbox
import torch
import torch.nn as nn
import numpy as np
from particul.od2.model import ClassifierWithParticulOD2
from particul.od2.fnrd.fnrd import FNRDModel
from torch_toolbox.datasets.dataset import (
    dataloader_add_parser_options,
    dataloaders_from_parser,
)
from tqdm import tqdm
from scipy.stats import spearmanr
from typing import Optional, Iterable, Callable
import argparse
from argparse import RawTextHelpFormatter


def get_dataset_confidence(
        model: nn.Module,
        reduction: Callable,
        dataset: Iterable,
        transform: str,
        intensity: float,
        device: Optional[str] = 'cpu',
        verbose: Optional[bool] = False,
) -> float:
    """ Compute confidence scores across a dataset

    :param model: Model
    :param dataset: Dataset
    :param reduction: Output reduction function
    :param transform: On-the-fly transform
    :param intensity: Transform intensity
    :param device: Target device (default: cpu)
    :param verbose: Verbose mode
    :returns: Average softmax predictions and confidence scores (shape: (1+P))
    """
    res = 0

    # Set evaluation mode
    model.eval()
    dataset = dataset if not verbose else tqdm(dataset, desc=f'{transform} {intensity}')

    for imgs, _ in dataset:
        # Apply on-the-fly operation (if applicable)
        if transform == "gaussian_noise":
            imgs = torch.stack([img + torch.randn(img.size()) * intensity * (torch.max(img) - torch.min(img))
                                for img in imgs])
        elif transform == "vertical_shift":
            imgs = torch.stack([img.roll(shifts=int(intensity), dims=1) for img in imgs])
        elif transform == "horizontal_shift":
            imgs = torch.stack([img.roll(shifts=int(intensity), dims=2) for img in imgs])

        # Move images to device
        imgs = imgs.to(device, non_blocking=True)

        # Compute prediction and confidence scores
        outputs = model(imgs)
        if reduction is not None:
            # Output reduction
            outputs = reduction(outputs)
        assert outputs.dim() == 1, "Invalid output dimension"
        outputs = outputs.detach().cpu().numpy()
        res += np.mean(outputs)

    return res/len(dataset)


def select_part_scores(x: torch.Tensor):
    return x[1]


def reduce_part_scores(x: torch.Tensor):
    # Average scores across detectors
    return torch.mean(select_part_scores(x), dim=1)


def reduce_softmax(x: torch.Tensor):
    # Average scores across detectors
    return torch.max(torch.softmax(x, dim=1), dim=1)[0]


supported_transforms = ['rotation', 'gaussian_blur', 'gaussian_noise',
                        'brightness', 'contrast', 'saturation', 'vertical_shift', 'horizontal_shift',
                        'none']


def _main():
    ap = argparse.ArgumentParser(
        description='Evaluate a confidence measure against perturbations on a given dataset.',
        formatter_class=RawTextHelpFormatter)
    subparsers = ap.add_subparsers(help='Select type of model')
    parser_classifier_with_particul = subparsers.add_parser('classifier-with-detectors',
                                                            help='Model is a classifier with Particul-OD2 module')
    parser_classifier_with_particul.set_defaults(which='classifier-with-detectors')
    ClassifierWithParticulOD2.add_parser_options(parser_classifier_with_particul)
    parser_particul = subparsers.add_parser('detectors-only',
                                            help='Model is a Particul detector')
    parser_particul.add_argument('-m', '--model', type=str, required=True,
                                 metavar='<path_to_file>',
                                 help='Path to calibrated model.')
    parser_particul.set_defaults(which='detectors')
    parser_classifier = subparsers.add_parser('classifier-only', help='Model is a classifier')
    parser_classifier.add_argument('-m', '--model', type=str, required=True, metavar='<path_to_file>',
                                   help='Path to trained model.')
    parser_classifier.set_defaults(which='classifier')
    parser_fnrd = subparsers.add_parser('classifier-probed', help='Model is a probed classifier')
    FNRDModel.add_parser_options(parser_fnrd)
    parser_fnrd.set_defaults(which='classifier-probed')
    ap.add_argument('--transform', type=str, required=True,
                    choices=supported_transforms,
                    metavar='<name>',
                    help='Transformation name.')
    ap.add_argument('--values', type=float, required=False, nargs='+',
                    metavar='<val>',
                    default=[0],
                    help='Transformation intensities.')
    ap.add_argument('--log', type=str, required=True,
                    metavar='<path_to_file>',
                    help='Path to output file.')
    ap.add_argument('--device', type=str, required=False,
                    metavar='<device_id>',
                    default='cpu',
                    help='Target device for execution.')
    ap.add_argument('-v', "--verbose", action='store_true',
                    help="Verbose mode")

    # Add dataset options
    dataloader_add_parser_options(ap)
    args = ap.parse_args()
    verbose = args.verbose
    args.verbose = False

    # Load model
    if args.which == 'classifier-with-detectors':
        model = ClassifierWithParticulOD2.build_from_parser(args)
        model.mode = 'production'
        reduction = select_part_scores
    elif args.which == 'classifier':
        model = torch_toolbox.load(args.model, map_location=args.device)
        reduction = reduce_softmax
    elif args.which == 'classifier-probed':
        model = FNRDModel.build_from_parser(args)
        reduction = select_part_scores
    else:  # Particul detector
        model = torch_toolbox.load(args.model, map_location=args.device)
        model.enable_confidence = True
        model.enable_features = False
        reduction = reduce_part_scores

    # Keep track of basic preprocessing
    default_preprocessing = args.preprocessing

    stats = np.zeros((len(args.values), 2))

    for index, val in enumerate(args.values):
        # Enrich preprocessing
        op = ""
        if args.transform == 'rotation':
            op = f'RandomRotation(degrees=[{val},{val}])+'
        elif args.transform == 'gaussian_blur':
            op = f'GaussianBlur(kernel_size=3, sigma={val})+' if val > 0 else ''
        elif args.transform == 'brightness':
            op = f'ColorJitter(brightness=[{val},{val}])+'
        elif args.transform == 'contrast':
            op = f'ColorJitter(contrast=[{val},{val}])+'
        elif args.transform == 'saturation':
            op = f'ColorJitter(saturation=[{val},{val}])+'
        args.preprocessing = [op + p for p in default_preprocessing]
        # Load dataset and compute average confidence and softmax scores
        _, testset, _ = dataloaders_from_parser(args)
        score = get_dataset_confidence(model, reduction, testset, args.transform, val, args.device, verbose)
        stats[index] = [val, score]

    # Save stats
    with open(args.log, 'a') as fout:
        header = f'{args.transform}; Avg_confidence; '
        np.savetxt(fout, stats, delimiter=";", header=header)

    # Compute spearman rank correlation scores
    print(stats)
    print('Spearman: ', spearmanr(stats[:, 0], stats[:, 1]))


if __name__ == '__main__':
    _main()
