import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as torch_datasets
from typing import Optional, Callable, List, Dict, Tuple
from .preprocessing import (
    preprocessing_add_parser_options,
    preprocessing_from_parser,
    preprocessing_from_string,
)
from .tardatasets import MultiImageArchive_build_from_config
from argparse import ArgumentParser, Namespace
from itertools import repeat
import os

# Temporary while waiting for stable support of these datasets
from .cub200 import CUB200
from .celeba_blurry import CelebABlurry, CelebANonBlurry


def _read_config(
        filename: str,
        mandatory: List[str],
        optional: Optional[List[str]] = None,
) -> Dict:
    """ Given a file containing lines in the form "field: value",
    return the corresponding dictionnary.
    Note: Lines starting ith "#" are ignored
    :param filename: Path to file
    :param mandatory: List of mandatory fields
    :param optional: List of optional fields
    :return: Configuration dictionnary
    """
    if optional is None:
        optional = []
    config = {}
    with open(filename, 'r') as fin:
        lines = [line.lstrip() for line in fin.read().splitlines()]
        for line in lines:
            # Skip comments
            if line.startswith('#'):
                continue
            tokens = [t.strip() for t in line.split(':')]
            if len(tokens) != 2:
                raise ValueError(f'Invalid configuration format: {line}')
            [key, value] = tokens
            if key not in mandatory + optional:
                raise ValueError(f'Unknown field : {key}')
            config[key] = value
    for n in mandatory:
        if n not in config.keys():
            raise ValueError(f'Missing mandatory field: {n}')
    return config


def get_classification_dataset(
        name: str,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: Optional[bool] = True,
        verbose: Optional[bool] = False,
) -> Dataset:
    """  Load a classification dataset

    :param name: Dataset or configuration name. Supported datasets are
        CIFAR10, CIFAR100, Caltech101, CelebA, FGVCAircraft, StanfordCars, CUB200, MNIST, FashionMNIST, SVHN,
        CelebABlurry, CelebANonBlurry.
        Otherwise, name is assumed to be an existing MultiImageArchive configuration file
        located in the root directory.
    :param root: Root directory of dataset
    :param split: Dataset split name. Either 'train','valid' or 'test'
    :param transform: A function/transform that takes in an PIL image
        and returns a transformed version.
    :param target_transform: A function/transform that takes in the
        target and transforms it.
    :param download: Download dataset if it is not present in the root
            directory.
    :param verbose: Verbose mode
    :return:
        An iterable dataset returning batches of tuples (images, labels)
    """
    supported_datasets = [
        'CIFAR10',
        'CIFAR100',
        'Caltech101',
        'CelebA',
        'CelebABlurry',
        'CelebANonBlurry',
        'FGVCAircraft',
        'StanfordCars',
        'CUB200',
        'FashionMNIST',
        'MNIST',
        'SVHN',
    ]
    dataset = None
    if name in supported_datasets:
        if name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']:
            if split == "trainval":
                split = "train"
            if split not in ['train', 'test']:
                raise ValueError(f'WARNING: Unsupported split {split} for dataset {name}')
            dataset = torch_datasets.__dict__[name](
                root=root,
                train=(split == 'train'),
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
        elif name == 'CelebA':
            dataset = torch_datasets.__dict__[name](
                root=root,
                split=split,
                target_type='identity',
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
        elif name == 'FGVCAircraft':
            if split == 'valid':
                split = 'val'
            dataset = torch_datasets.__dict__[name](
                root=root,
                split=split,
                annotation_level='variant',
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
        elif name in ['StanfordCars', 'SVHN']:
            if split is None:
                split = 'train'
            if split not in ['train', 'test']:
                raise ValueError(f'WARNING: Unsupported split {split} for dataset {name}')
            dataset = torch_datasets.__dict__[name](
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
        elif name == 'Caltech101':
            dataset = torch_datasets.__dict__[name](
                root=root,
                target_type='category',
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
            dlen = len(dataset)
            lengths = [int(dlen * 0.7), int(dlen * 0.15), dlen - int(dlen * 0.7) - int(dlen * 0.15)]
            trainset, testset, valset = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
            if split == 'train':
                dataset = trainset
            elif split == 'test':
                dataset = testset
            elif split == 'valid':
                dataset = valset
        elif name == 'CUB200':
            if split == 'trainval':
                split = 'train'
            if split not in ['train', 'test', None]:
                raise ValueError(f'WARNING: Unsupported split {split} for dataset {name}')
            dataset = CUB200(
                root=root,
                split=split,
                transform=transform,
                download=download,
            )
        elif name == "CelebABlurry":
            dataset = CelebABlurry(
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
        elif name == "CelebANonBlurry":
            dataset = CelebANonBlurry(
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )

    else:
        dataset = MultiImageArchive_build_from_config(
            config_file=os.path.join(root, f'{name}.txt'),
            split=split,
            transform=transform,
            target_transform=target_transform,
        )
    if verbose:
        print(f'{split} size: {len(dataset)}')
        if transform:
            print(f'> Image preprocessing configuration: \n {transform}')
    return dataset


def dataset_add_parser_options(parser: ArgumentParser) -> None:
    """ Add all common arguments for the init of dataset to an existing parser.

    :param parser: argparse argument parser.
    """
    preprocessing_add_parser_options(parser)
    parser.add_argument("--dataset-name", required=True, type=str,
                        metavar='<name>',
                        help="Dataset name")
    parser.add_argument("--dataset-location", required=True, type=str,
                        metavar='<path>',
                        help="Dataset location")
    parser.add_argument("--dataset-download", required=False, action='store_true',
                        help="Download dataset if necessary")


def datasets_from_parser(args: Namespace) -> Tuple[Optional[Dataset], ...]:
    """ Load train/test/val datasets from a parser output

    :param args: Parser output
    :return: train, test and valid datasets if possible
    """
    transform, target_transform = preprocessing_from_parser(args, broadcast_to=3)
    datasets = []
    verbose = hasattr(args, 'verbose') and args.verbose
    for t, name in zip(transform, ['train', 'test', 'valid']):
        dataset = None
        try:
            dataset = get_classification_dataset(
                name=args.dataset_name,
                root=args.dataset_location,
                split=name,
                transform=t,
                target_transform=target_transform,
                download=args.dataset_download,
                verbose=verbose,
            )
        except ValueError:
            pass
        datasets.append(dataset)
    return tuple(datasets)


def dataset_from_file(
        path: str,
        verbose: Optional[bool] = False,
) -> Dataset:
    """ Load dataset from a configuration file containing the mandatory fields:

    The configuration file is a plaintext file containing the following mandatory fields:

    - ``dataset_location``: Path to dataset root directory.
    - ``dataset_name``: Dataset name or configuration.
    - ``dataset_split``: ``train``, ``test`` or ``valid``
    - ``preprocessing``: Preprocessing string.
    - ``normalization``: Normalization function.

    :param path: Path to file
    :param verbose: Verbose mode
    :return: Dataset
    """
    mandatory_keys = [
        'dataset_name',
        'dataset_location',
        'dataset_split',
        'preprocessing',
        'normalization',
    ]
    config = _read_config(path, mandatory_keys)
    transform = preprocessing_from_string(
        config['preprocessing'], config['normalization'])
    return get_classification_dataset(
        name=config['dataset_name'],
        root=config['dataset_location'],
        split=config['dataset_split'],
        transform=transform,
        target_transform=None,
        download=True,
        verbose=verbose,
    )


# Useful collate functions when dealing with unusual datasets
def collate_NI1L(data: List) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate B tuples of (N images, 1 label) into tuple of (BxN images, B labels)

    :param data: List of len == batch_size where each entry is a tuple
        ([img1,...,imgN], [label])
    :return: Tuple of (BxN images, B labels)
    """
    imgs = torch.stack([img for e in data for img in e[0]])
    labels = torch.stack([torch.tensor(int(label)) for e in data for label in e[1]])
    return imgs, labels


def collate_NINL(data: List) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate B tuples of (N images, 1 label) into tuple of (BxN images, BxN labels)

    :param data: List of len == batch_size where each entry is a tuple
        ([img1,...,imgN], [label])
    :return: Tuple of (BxN images, BxN labels)
    """
    n = len(data[0][0])
    imgs = torch.stack([img for e in data for img in e[0]])
    labels = torch.stack([torch.tensor(int(label)) for e in data for label in repeat(e[1], n)])
    return imgs, labels


collate_fns = {
    'NI1L': collate_NI1L,
    'NINL': collate_NINL,
}


def dataloader_add_parser_options(parser: ArgumentParser) -> None:
    """ Add all common arguments for the init of a dataloader to an existing parser.

    :param parser: argparse argument parser.
    """
    dataset_add_parser_options(parser)
    parser.add_argument("--dataloader-shuffle", required=False, action='store_true',
                        help="Shuffle dataset")
    parser.add_argument("--batch-size", required=False, type=int,
                        metavar='<val>',
                        default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--dataloader-collate", required=False, type=str,
                        metavar='<func>',
                        choices=['NI1L', 'NINL'],
                        help="Dataloader collate function. Choices: NI1L/NINL")


def dataloaders_from_parser(args: Namespace) -> Tuple[Optional[DataLoader], ...]:
    """ Load train/test/val dataloaders from a parser output

    :param args: Parser output
    :return: train, test and valid dataloader if possible
    """
    datasets = datasets_from_parser(args)
    collate_fn = collate_fns[args.dataloader_collate] if args.dataloader_collate else None

    return tuple([
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.dataloader_shuffle,
            collate_fn=collate_fn,
        ) if dataset else None for dataset in datasets])


def dataloader_from_file(
        path: str,
        batch_size: int,
        shuffle: Optional[bool] = False,
        verbose: Optional[bool] = False,
) -> DataLoader:
    """ Load dataloader from a configuration file.

    The configuration file is a plaintext file containing the following mandatory fields:

    - ``dataset_location``: Path to dataset root directory.
    - ``dataset_name``: Dataset name or configuration.
    - ``dataset_split``: ``train``, ``test`` or ``valid``
    - ``preprocessing``: Preprocessing string.
    - ``normalization``: Normalization function.

    :param path: Path to file
    :param batch_size: Batch size
    :param shuffle: Shuffle data
    :param verbose: Verbose mode
    :return: Dataloader
    """
    dataset = dataset_from_file(path, verbose)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
