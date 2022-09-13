#!/usr/bin/env python3
from particul.od2.model import ClassifierWithParticulOD2
import torch_toolbox
from torch_toolbox.datasets.dataset import (
    get_classification_dataset,
    preprocessing_add_parser_options,
    preprocessing_from_parser,
)
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import torch
from particul.od2.fnrd.fnrd import FNRDModel
from numpy import ndarray, savetxt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, List, Callable
from progress.bar import Bar
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.neighbors import KernelDensity
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def kde_curve (X: np.array, nsamples: int = 1000) -> np.array:
    """ Given a 1D distribution of values, return kernel density estimation curve
    """
    if X.ndim < 2:
        X = X[:, np.newaxis]
    elif X.ndim == 2:
        assert X.shape[1] == 1, f'Invalid data shape {X.shape}'
    else:
        assert False, f'Invalid data shape {X.shape}'
    kde = KernelDensity(kernel='gaussian',bandwidth=0.2).fit(X)
    delta = np.max(X)-np.min(X)
    X_plot = np.linspace(np.min(X)-delta*0.5,np.max(X)+delta*0.5,nsamples)[:, np.newaxis]
    Y_plot = kde.score_samples(X_plot)[:, np.newaxis]
    return np.concatenate([X_plot, Y_plot], axis=1)

def detector_quality(
        detector: nn.Module,
        ind_dataset: Dataset,
        ind_name: str,
        ood_datasets: List[Dataset],
        ood_names: List[str],
        save_prefix: str,
        batch_size: int,
        reduction: Optional[Callable] = None,
        device: Optional[str] = 'cpu',
        verbose: Optional[bool] = False,
        plot: Optional[bool] = False,
) -> Tuple[List[float], List[float], List[float]]:
    """ Given a detector and a reference dataset, compute the quality of the detector by returning, for each given OoD
    dataset:

        - the Area Under the Receiver Operating Characteristic (AUROC) curve (true positive rate v. false positive rate)
        - the Area Under the Precision Recall (AUPR) curve (precision v. recall)
        - the FPR80 score (false positive rate when true positive rate is 80%)

    :param detector: Detector under test
    :param ind_dataset: In-distribution dataset
    :param ind_name: Name of in-distribution dataset
    :param ood_datasets: Out-of-Distribution datasets
    :param ood_names: Names of Out-of-Distribution datasets (used when plotting curves)
    :param save_prefix: Prefix to curve files (if None: do not save curves)
    :param batch_size: Batch size
    :param reduction: Output reduction function
    :param device: Target device
    :param verbose: Verbose mode
    :param plot: Plot curves
    :returns AUROC values, AUPR values, FPR80 values
    """
    def compute_outputs(dataset: Dataset, name: str) -> ndarray:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
        res = None
        bar = Bar(f'Computing outputs on {name}',
                  max=len(dataloader),
                  suffix='%(percent).1f%%') if verbose else None

        for imgs, _ in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            outputs = detector(imgs)
            if reduction is not None:
                # Output reduction
                outputs = reduction(outputs)
            assert outputs.dim() == 1, "Invalid output dimension"
            outputs = outputs.detach().cpu().numpy()
            res = np.concatenate([res, outputs]) if res is not None else outputs
            if bar:
                bar.next()
        if bar:
            bar.finish()
        return res

    # Compute reference outputs
    ind_outputs = compute_outputs(ind_dataset, "In-distribution")
    # Sort output values and determine threshold for TPR=80%
    ind_outputs = np.sort(ind_outputs)
    tr80 = ind_outputs[int(len(ind_outputs) * 0.2)]
    if save_prefix is not None:
        savetxt(f'{save_prefix}{ind_name}_kde.csv', kde_curve(ind_outputs, 1000), delimiter=',')

    # Prepare plot
    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[1].set_title(f'{ind_name} ROC curve')
        axs[1].set_xlabel('False positive rate')
        axs[1].set_ylabel('True positive rate')
        axs[0].set_title('Distributions')
        axs[0].set_xlabel('Confidence measure')
        axs[0].set_ylabel('Density')

    auroc_scores = []
    aupr_scores = []
    fpr80_scores = []

    for name, dataset, color in zip(ood_names, ood_datasets, mcolors.TABLEAU_COLORS):
        # Compute outputs on OoD dataset
        ood_outputs = compute_outputs(dataset, name)
        y_true = np.array([1]*len(ind_dataset) + [0]*len(dataset))
        y_score = np.concatenate([ind_outputs, ood_outputs])

        # Compute scores
        auroc_scores.append(roc_auc_score(y_true=y_true, y_score=y_score)*100)
        aupr_scores.append(average_precision_score(y_true=y_true, y_score=y_score)*100)
        fpr80_scores.append(np.sum(ood_outputs > tr80)/len(ood_outputs)*100)

        if save_prefix is not None:
            savetxt(f'{save_prefix}{name}_kde.csv', kde_curve(ood_outputs, 1000), delimiter=',')
            fpr, tpr, _ = roc_curve(y_true, y_score)
            savetxt(f'{save_prefix}{ind_name}_v_{name}_ROC.csv',np.swapaxes(np.array([fpr,tpr]), 0, 1), delimiter=',')
            prc, rec, _ = precision_recall_curve(y_true, y_score)
            savetxt(f'{save_prefix}{ind_name}_v_{name}_PR.csv',np.swapaxes(np.array([prc,rec]), 0, 1), delimiter=',')

        # Update plot
        if plot:
            axs[0].hist(ood_outputs, bins=100, density=True, alpha=0.5, label=f'{name} (OoD)')
            fpr, tpr, _ = roc_curve(y_true, y_score)
            axs[1].plot(fpr, tpr, color=color, label=f'{name}')

    if plot:
        # Insert InD histogram last so that all OoD curve/histogram color match.
        axs[0].hist(ind_outputs, bins=100, density=True, alpha=0.5, label=f'{ind_name} (InD)')
        axs[0].legend()
        axs[1].legend()
        plt.show()

    return auroc_scores, aupr_scores, fpr80_scores


def select_part_scores(x: torch.Tensor):
    return x[1]


def reduce_part_scores(x: torch.Tensor):
    # Average scores across detectors
    return torch.mean(select_part_scores(x), dim=1)


def reduce_softmax(x: torch.Tensor):
    # Average scores across detectors
    return torch.max(torch.softmax(x, dim=1), dim=1)[0]


def _main():
    ap = argparse.ArgumentParser(
        description='Evaluate a Particul-based Out-of-Distribution detector.',
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
    preprocessing_add_parser_options(ap)
    ap.add_argument('--ind-name', type=str, required=True,
                    metavar='<name>',
                    help='In-distribution dataset name.')
    ap.add_argument('--ind-location', type=str, required=True,
                    metavar='<path_to_directory>',
                    help='In-distribution dataset location.')
    ap.add_argument('--ood-name', type=str, required=True, nargs='+',
                    metavar='<name>',
                    help='Out-of-distribution dataset names.')
    ap.add_argument('--ood-location', type=str, required=True, nargs='+',
                    metavar='<path_to_directory>',
                    help='Out-of-distribution datasets locations.')
    ap.add_argument('--batch-size', type=int, required=False,
                    metavar='<val>',
                    default=8,
                    help='Batch size.')
    ap.add_argument('--save-curves', type=str, required=False,
                    metavar='<file_prefix>',
                    default=None,
                    help='Save curves into files.')
    ap.add_argument('--device', type=str, required=False,
                    metavar='<device_id>',
                    default='cpu',
                    help='Target device for execution.')
    ap.add_argument('-p', "--plot", action='store_true',
                    help="Plot curves")
    ap.add_argument('-v', "--verbose", action='store_true',
                    help="Verbose mode")
    args = ap.parse_args()

    if len(args.ood_location) != 1 and len(args.ood_location) != len(args.ood_name):
        ap.error(f'Invalid number of OoD dataset locations')
    if len(args.ood_location) != len(args.ood_name):
        args.ood_location = args.ood_location * len(args.ood_name)

    input_ops, target_ops = preprocessing_from_parser(args)
    ind_dataset = get_classification_dataset(
        name=args.ind_name,
        root=args.ind_location,
        split="test",
        transform=input_ops[0],
        target_transform=target_ops,
        download=False,
        verbose=args.verbose,
    )
    ood_datasets = [get_classification_dataset(
        name=name,
        root=location,
        split="test",
        transform=input_ops[0],
        target_transform=target_ops,
        download=False,
        verbose=args.verbose,
    ) for name, location in zip(args.ood_name, args.ood_location)]

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
    else:
        model = torch_toolbox.load(args.model, map_location=args.device)
        model.enable_confidence = True
        model.enable_features = False
        reduction = reduce_part_scores
    model.eval()

    auroc_scores, aupr_scores, fpr80_scores = detector_quality(
        detector=model,
        ind_dataset=ind_dataset,
        ind_name=args.ind_name,
        ood_datasets=ood_datasets,
        ood_names=args.ood_name,
        save_prefix=args.save_curves,
        batch_size=args.batch_size,
        reduction=reduction,
        device=args.device,
        verbose=args.verbose,
        plot=args.plot,
    )
    for name, auroc, aupr, fpr80 in zip(args.ood_name, auroc_scores, aupr_scores, fpr80_scores):
        print(f'{args.ind_name} v. {name}: AUROC={auroc:.1f}, AUPR={aupr:.1f}, FPR80={fpr80:.1f}')


if __name__ == '__main__':
    _main()
