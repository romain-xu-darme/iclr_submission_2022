import torch
from torchvision.transforms import *
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Callable, Optional, Any, Dict, Union
import numpy as np

# Common datasets normalization
datasets_norms = {
    'imagenet': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'CIFAR10': Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'MNIST': Normalize((0.1307,), (0.3081,)),
    'FMNIST': Normalize((0.2860,), (0.3530,)),
    'None': None,
}

# "Common" datasets preprocessing
common_preprocessing = {
    'CIFAR10': [
        RandomCrop(32, padding=4, padding_mode="reflect"),
        RandomHorizontalFlip(),
    ],
    'Geometric': [
        RandomAffine(degrees=2.0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=1.0),
        ColorJitter(brightness=(0.8, 1.0)),
        RandomHorizontalFlip(p=0.5),
    ],
    'RandCrop448': [
        RandomResizedCrop(448, scale=(0.8, 1.0)),
        RandomHorizontalFlip(),
    ],
    'CenterCrop448': [
        Resize((512, 512)),
        CenterCrop(448),
    ],
    'Resize448': [
        Resize((448, 448)),
    ],
    'Resize224': [
        Resize((224, 224)),
    ],
    # InceptionV3 requires inputs of shape 299x299
    'RandCrop299': [
        RandomResizedCrop(448, scale=(0.8, 1.0)),
        RandomHorizontalFlip(),
    ],
    'CenterCrop299': [
        Resize((342, 342)),
        CenterCrop(299),
    ],
    'Resize299': [
        Resize((299, 299)),
    ],
    'ToTensor': [
        ToTensor(),
    ],
}


def _dictlist2string(d: Dict) -> str:
    res = ""
    for n in d:
        res += f'\t- {n}:\n'
        for f in d[n]:
            res += f'\t\t + {f}\n'
    return res


def preprocessing_add_parser_options(parser: ArgumentParser) -> None:
    """ Add all common arguments for the preprocessing functions to an existing parser.

    :param parser: argparse argument parser.
    """
    parser.add_argument('--preprocessing', type=str, required=False,
                        metavar=('<train_config>', '<test_config> [<val_config>]'), nargs='+',
                        default=[""],
                        help='Specify preprocessing functions for each dataset (default: None).\n' \
                             'Each set of functions should be linked by "+", parenthesis escaped.\n' \
                             'e.g. --preprocessing RandomCrop\(\(448,448\)\)+RandomHorizontalFlip\(\) ' \
                             'CenterCrop448\n' \
                             '                     |----------  Trainset preprocessing -------------| ' \
                             '|---- Testset-- ---|\n' \
                             'List of pre-defined common transforms: \n'
                             f'{_dictlist2string(common_preprocessing)}'
                        )
    parser.add_argument('--normalization', type=str, required=False,
                        metavar='<function>',
                        choices=datasets_norms.keys(),
                        default='imagenet',
                        help=f'Set dataset normalization function.')
    parser.add_argument('--target-preprocessing', type=str, required=False, nargs='+',
                        metavar='<config>',
                        choices=['ShiftCorrection'],
                        default=None,
                        help='Specify target preprocessing functions (default: None).')
    parser.add_argument('--seed', type=int, required=False,
                        metavar='<seed>',
                        default=None,
                        help='Use random seed')


def preprocessing_from_string(
        transforms: str,
        normalization: str,
) -> Callable:
    """ Given a transform string and the name of a normalization function,
    returns the corresponding preprocessing function.

    :param transforms: Transform string
    :param normalization: Normalization function name
    :return: Preprocessing function

    A transform string is a list of operations separated by a ``+`` symbol.
    Each operation is either :

    - an explicit call to one of the following functions from \
            `torchvision.transforms <https://pytorch.org/vision/0.9/transforms.html>`_ :

        - ``CenterCrop``, ``ColorJitter``, ``Compose``, ``GaussianBlur``, ``Normalize``,\
        ``RandomAffine``, ``RandomCrop``, ``RandomHorizontalFlip``, \
        ``RandomResizedCrop``, ``Resize``, ``ToTensor``
        - In that case, all parenthesis must be escaped in the command line, e.g.

        "Resize\\\(\\\(224,224\\\)\\\)+ToTensor\\\(\\\)"

    - or a predefined function among the following:
        - ``CIFAR10``:
            - RandomCrop(size=(32, 32), padding=4)
            - RandomHorizontalFlip(p=0.5)
        - ``Geometric``:
            - RandomAffine(degrees=[-3.0, 3.0], translate=(0.1, 0.1), scale=(0.9, 1.1), shear=[-2.0, 2.0])
            - ColorJitter(brightness=(0.8, 1.0), contrast=None, saturation=None, hue=None)
            - RandomHorizontalFlip(p=0.5)
        - ``RandCrop448``:
            - RandomResizedCrop(size=(448, 448), scale=(0.8, 1.0), ratio=(0.75, 1.3333), interpolation=bilinear)
            - RandomHorizontalFlip(p=0.5)
        - ``CenterCrop448``:
            - Resize(size=(512, 512), interpolation=bilinear, max_size=None, antialias=None)
            - CenterCrop(size=(448, 448))
        - ``Resize448``:
            - Resize(size=(448, 448), interpolation=bilinear, max_size=None, antialias=None)
        - ``RandCrop299:``
            - RandomResizedCrop(size=(448, 448), scale=(0.8, 1.0), ratio=(0.75, 1.3333), interpolation=bilinear)
            - RandomHorizontalFlip(p=0.5)
        - ``CenterCrop299``:
            - Resize(size=(342, 342), interpolation=bilinear, max_size=None, antialias=None)
            - CenterCrop(size=(299, 299))
        - ``Resize299``:
            - Resize(size=(299, 299), interpolation=bilinear, max_size=None, antialias=None)
        - ``ToTensor``:
            - ToTensor()

    The normalization string can be either:

        - ``imagenet``: Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        - ``CIFAR10``:  Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        - ``MNIST``:    Normalize((0.1307,), (0.3081,))
        - ``FMNIST``:   Normalize((0.1307,), (0.3081,))
        - ``None``: None

    """
    ops = []
    for func in transforms.split('+'):
        if func == 'None':
            break
        if func in common_preprocessing.keys():
            op = common_preprocessing[func]
        # Deprecated: this is very dangerous from a security point of view!!!
        else:
            op = eval(func)
        op = op if type(op) == list else [op]
        ops += op
    normalize = datasets_norms[normalization]
    if normalize:
        ops.append(normalize)
    return Compose(ops) if len(ops) > 1 else ops


def preprocessing_from_parser(
        args: Namespace,
        broadcast_to: Optional[int] = None
) -> Tuple[List[Callable[..., Any]], Callable[[Any], Union[int, Any]]]:
    """ Return preprocessing function from output of a parser

    :param args: Result from argparse
    :param broadcast_to: Broadcast operations if necessary
    :return: tuple of preprocessing functions
    """
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Build list of image transformation operations
    ops = [preprocessing_from_string(p, args.normalization) for p in args.preprocessing]

    if broadcast_to and (len(ops) != broadcast_to):
        if len(ops) == 1:
            ops = [ops[0]] * broadcast_to
        elif len(ops) < broadcast_to:
            ops += [None] * (broadcast_to - len(ops))
        else:
            raise ValueError(f'Too many transforms provided {len(ops)}. '
                             f'Expected {broadcast_to}.')

    if not args.target_preprocessing:
        return ops, None

    def target_preprocessing(target: Any):
        if 'ShiftCorrection' in args.target_preprocessing:
            target = int(target) - 1
        return target

    return ops, target_preprocessing


def split_transform(ops: Callable) -> Tuple[Callable, Callable, Callable]:
    """ Split operations into transform / reshape / preprocess phases

    :param ops: Operations
    :return: transform, reshape, preprocessing

    Note that reshape and preprocessing functions are reproducible, while transform may
    contain random operations (including RandomCrop and RandomResizedCrop)
    """
    if isinstance(ops, Compose):
        ops = ops.transforms
        for i, t in reversed(list(enumerate(ops))):
            if isinstance(t, (Resize, CenterCrop)):
                trans_ops, shape_ops, prepr_ops = ops[:i], ops[i], ops[i + 1:]
                trans_ops = Compose(trans_ops) if len(trans_ops) > 1 else trans_ops[0] if trans_ops else None
                prepr_ops = Compose(prepr_ops) if len(prepr_ops) > 1 else prepr_ops[0] if prepr_ops else None
                return trans_ops, shape_ops, prepr_ops
            elif isinstance(t, (RandomCrop, RandomResizedCrop)):
                trans_ops, shape_ops, prepr_ops = ops[:i + 1], None, ops[i + 1:]
                trans_ops = Compose(trans_ops) if len(trans_ops) > 1 else trans_ops[0] if trans_ops else None
                prepr_ops = Compose(prepr_ops) if len(prepr_ops) > 1 else prepr_ops[0] if prepr_ops else None
                return trans_ops, shape_ops, prepr_ops

    elif isinstance(ops, (Resize, CenterCrop)):
        return None, ops, None
    elif isinstance(ops, (RandomCrop, RandomResizedCrop)):
        return ops, None, None
    raise ValueError('Missing Resize() function from preprocessing.')
