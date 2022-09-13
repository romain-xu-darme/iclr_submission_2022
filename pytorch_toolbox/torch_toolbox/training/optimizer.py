import torch
import torch.nn as nn
import ast
from typing import Tuple, Dict, List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from argparse import ArgumentParser, Namespace

default_opt_configs = {
    'Adam': {
        'lr': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-08,
        'decay': 0.0
    },
    'SGD': {
        'lr': 0.01,
        'momentum': 0.0,
        'decay': 0.0,
        'nesterov': False
    },
    'RMSprop': {
        'lr': 0.001,
        'alpha': 0.9,
        'epsilon': 1e-08,
        'decay': 0.0
    }
}

default_sch_configs = {
    'StepLR': {
        'step': 30,
        'gamma': 0.1,
    },
    'ReduceLROnPlateau': {
        'mode': 'min',
        'factor': 0.1,
        'patience': 5,
        'threshold': 1e-4,
    },
}


def optimizer_add_parser_options(parser: ArgumentParser) -> None:
    """ Add optimizer options to a argparse parser

    :param parser: Existing Argparse parser
    """
    parser.add_argument("--optimizer", required=True, type=str, nargs='+',
                        metavar=('<name>', '<params>'),
                        help='Training optimizer.\n'
                             '\t - Adam PARAM=<value> where PARAM is:\n'
                             '\t     + lr (default: 1e-3)\n'
                             '\t     + beta_1 (default: 0.9)\n'
                             '\t     + beta_2 (default: 0.999)\n'
                             '\t     + epsilon (default: 1e-8)\n'
                             '\t     + decay (default: 0)\n'
                             '\t - SGD PARAM=<value> where PARAM is:\n'
                             '\t     + lr (default: 1e-2)\n'
                             '\t     + momentum (default: 0)\n'
                             '\t     + decay (default: 0)\n'
                             '\t     + nesterov (default: False)\n'
                             '\t - RMSprop PARAM=<value> where PARAM is:\n'
                             '\t     + lr (default: 1e-3)\n'
                             '\t     + alpha (default: 0.9)\n'
                             '\t     + epsilon (default: 1e-8)\n'
                             '\t     +decay (default: 0)\n')
    parser.add_argument("--lr-scheduler", required=False, type=str, nargs='+',
                        metavar=('<name>', '<params>'),
                        help='Learning rate scheduler.\n'
                             '\t - StepLR PARAM=<value> where PARAM is:\n'
                             '\t     + step (default: 30)\n'
                             '\t     + gamma (default: 0.1)\n'
                             '\t - ReduceLROnPlateau PARAM=<value> where PARAM is;\n'
                             '\t     + mode (default:min) \n'
                             '\t     + factor (default: 0.1) \n'
                             '\t     + patience (default: 5) \n'
                             '\t     + threshold (default: 1e-4) \n')


def optimizer_from_parser(
        model: nn.Module,
        args: Namespace,
) -> Tuple[Optimizer, object]:
    """ Given the result of argparse, returns an optimizer and a learning rate scheduler

    An optimizer is configured according to the following format: ::

    <name> <param>=<value> [<param>=<value> ...]

    List of supported optimizers and parameters:

    - ``Adam``:
        - ``lr`` (default: 1e-3)
        - ``beta_1`` (default: 0.9)
        - ``beta_2`` (default: 0.999)
        - ``epsilon`` (default: 1e-8)
        - ``decay`` (default: 0)
    - ``SGD``:
        - ``lr`` (default: 1e-2)
        - ``momentum`` (default: 0)
        - ``decay`` (default: 0)
        - ``nesterov`` (default: False)
    - ``RMSprop``:
        - ``lr`` (default: 1e-3)
        - ``alpha`` (default: 0.9)
        - ``epsilon`` (default: 1e-8)
        - ``decay`` (default: 0)

    The learning rate scheduler is configured according to the following format: ::

    <name> <param>=<value> [<param>=<value> ...]

    List of supported schedulers and parameters:

    - ``StepLR`` (multiplies current LR by <gamma> every <step> epochs):
        - ``step`` (default: 30)
        - ``gamma`` (default: 0.1)
    - ``ReduceLROnPlateau`` (multiplies current LR by <factor> when detecting a \
            plateau (variation lower than <threshold>) during <patience> epochs:
        - ``mode`` (default:min)
        - ``factor`` (default: 0.1)
        - ``patience`` (default: 5)
        - ``threshold`` (default: 1e-4)


    :param model: Path to target model
    :param args: Result of argparse
    :return: Optimizer, learning rate scheduler
    """
    # Check optimizer name and recover default configuration
    opt_name = args.optimizer[0]
    if opt_name not in default_opt_configs:
        raise ValueError(f'Error: Unsupported optimizer {opt_name}')

    def update_config(config: Dict, params: List[str]) -> Dict:
        # Update configuration
        for a in params:
            if len(a.split('=')) != 2:
                raise ValueError(f'Invalid option: {a}')
            key = a.split('=')[0]
            val = a.split('=')[1]
            # Check that option exists
            if key not in config.keys():
                raise KeyError(f'[{opt_name}] {key} is not a valid option')
            # Cast value according to default option
            if isinstance(config[key], bool):
                config[key] = (val in ['True', 'true'])
            elif isinstance(config[key], float):
                config[key] = float(val)
            elif isinstance(config[key], int):
                config[key] = int(val)
            elif isinstance(config[key], str):
                config[key] = val
            elif isinstance(config[key], list):
                config[key] = ast.literal_eval(val)
            elif config[key] is None:
                config[key] = val
            else:
                raise KeyError(f'[{opt_name}] {key} type not supported: {str(type(config[key]))}')
        return config

    config = update_config(default_opt_configs[opt_name], args.optimizer[1:])

    optimizer = None
    if opt_name == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=config['lr'],
            betas=(config['beta_1'], config['beta_2']),
            eps=config['epsilon'],
            weight_decay=config['decay']
        )
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['decay'],
            nesterov=config['nesterov']
        )
    if opt_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            params=model.parameters(),
            lr=config['lr'],
            alpha=config['alpha'],
            eps=config['epsilon'],
            weight_decay=config['decay']
        )
    # Map optimizer to device
    if hasattr(args, 'device'):
        _optimizer_map_to_device(optimizer, args.device)

    # Learning rate scheduler
    scheduler = None
    if args.lr_scheduler is not None:
        sch_name = args.lr_scheduler[0]
        if sch_name not in default_sch_configs:
            raise ValueError(f'Error: Unsupported scheduler {sch_name}')
        config = update_config(default_sch_configs[sch_name], args.lr_scheduler[1:])
        if sch_name == 'StepLR':
            scheduler = StepLR(
                optimizer,
                step_size=config['step'],
                gamma=config['gamma'],
            )
        if sch_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=config['mode'],
                factor=config['factor'],
                patience=config['patience'],
                threshold=config['threshold'],
            )
    return optimizer, scheduler


def _optimizer_map_to_device_rec(state: dict, device: str) -> dict:
    """ Recursive function fixing device map in a dictionnary

    :param state: Target dictionnary
    :param device: Target device
    :return: Updated dictionary
    """
    for key in state.keys():
        var = state[key]
        if type(var) is dict:
            state[key] = _optimizer_map_to_device_rec(var, device)
        elif type(var) is torch.Tensor:
            state[key] = var.to(device)
    return state


def _optimizer_map_to_device(opt: Optimizer, device: str) -> None:
    """ Fix device map for optimizer

    :param opt: Optimizer
    :param device: Target device
    """
    opt.state_dict()['state'] = \
        _optimizer_map_to_device_rec(opt.state_dict()['state'], device)
