import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Iterable, Optional
from torchvision.transforms import *
from torch_toolbox.datasets.preprocessing import (
        datasets_norms,
        common_preprocessing,
)
from torch_toolbox.datasets.dataset import get_classification_dataset

def _read_config(path: str) -> Dict:
    """ Read/parse configuration file and return configuration dictionnary
    :param path: Path to configuration file
    :return: Configuration dictionnary
    """
    mandatory_keys = [
        'dataset_name',
        'dataset_path',
        'preprocessing',
        'normalization',
        'model_path',
    ]
    optional_keys = [
        'model_state_path',
    ]
    config = {}
    with open(path,'r') as fin:
        lines = [l.lstrip() for l in fin.read().splitlines()]
        for l in lines:
            if l.startswith('#'): continue
            tokens = [t.strip() for t in l.split(':')]
            if len(tokens) != 2:
                raise ValueError(f'Invalid configuration format: {l}')
            [key, value] = tokens
            if key not in mandatory_keys+optional_keys:
                raise ValueError(f'Unknown field : {key}')
            config[key] = value
    for n in mandatory_keys:
        if n not in config.keys():
            raise ValueError(f'Missing mandatory field: {n}')
    return config

def load_config(
        config_path: str,
        batch_size: int,
        dataset_split: Optional[str] = 'test',
        device: Optional[str] = 'cpu',
        verbose: Optional[bool] = False,
) -> Tuple[nn.Module, Iterable] :
    """ Read and load a configuration file, returning a model and a dataloader.

    The configuration file is a plaintext file containing the following mandatory fields:

    - ``dataset_path``: Path to dataset root directory.
    - ``dataset_name``: Dataset name or configuration.
    - ``preprocessing``: Preprocessing string.
    - ``normalization``: Normalization function.
    - ``model_path``: Path to model.

    :param config_path: Path to configuration file
    :param batch_size: Dataloader batch size
    :param dataset_split: Dataset split name
    :param device: Target device for model
    :return: model, dataloader
    """
    config = _read_config(config_path)

    # Setup preprocessing. This part is almost a copy/paste of
    # torch_toolbox.datasets.preprocessing.preprocessing_from_parser
    ops = []
    for func in config['preprocessing'].split('+'):
        if func == 'None':
            break
        op = common_preprocessing[func] if func in common_preprocessing.keys() \
                else eval(func)
        op = op if type(op)==list else [op]
        ops += op
    ops.append(datasets_norms[config['normalization']])
    preprocessing = Compose(ops) if len(ops) > 1 else ops

    # Load dataset
    dataset = get_classification_dataset(
        name = config['dataset_name'],
        root = config['dataset_path'],
        split = dataset_split,
        transform = preprocessing,
        target_transform = None,
        download = True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size
    )

    # Load model
    model = torch.load(config['model_path'],map_location=device)
    if 'model_state_path' in config.keys():
        state = torch.load(config['model_state_path'],map_location=device)
        model.load_state_dict(state)

    if verbose:
        print(f'> Configuration file: {config_path}\n' \
                f'\t + Preprocessing: {preprocessing}\n' \
                f'\t + Dataset length: {len(dataset)}\n' \
                f'\t + Model: {model}' \
        )
    return model, dataloader
