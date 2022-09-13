import torch
import torch.nn as nn
from .stats import ProgressMeter, AverageMeter
import torch_toolbox.models.serialization as serialization
from .optimizer import (
    optimizer_add_parser_options,
    optimizer_from_parser
)
from .checkpoint import (
    checkpoint_add_parser_options,
    save_checkpoint,
    load_checkpoint
)
from torch_toolbox.datasets.dataset import (
    dataloader_add_parser_options,
    dataloaders_from_parser
)
from pathlib import Path
import time

from typing import Optional, Iterable, Tuple, List, Any, Callable
from argparse import ArgumentParser, Namespace
from torch.optim import Optimizer


def train(
        dataset: Iterable,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        epoch: int,
        device: Optional[str] = 'cpu',
        verbose: Optional[bool] = False,
) -> List[float]:
    """ Train model on a dataset for one epoch

    :param dataset: Dataset
    :param model: Trained model
    :param optimizer: Learning optimizer
    :param loss_fn: Prototype loss and metrics
    :param epoch: Epoch index
    :param device: Target device
    :param verbose: Verbose mode
    :return: List of metrics
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    metrics_values = [AverageMeter(n, ':6.3f') for n in loss_fn.metrics]
    if verbose:
        progress = ProgressMeter(
            len(dataset),
            [batch_time, data_time] + metrics_values,
            prefix="Epoch: [{}]".format(epoch),
        )

    # Set model training mode
    model.train()

    end = time.time()
    # Main training loop
    for batch_index, (imgs, labels) in enumerate(dataset):
        # Map to device
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Data loading time
        data_time.update(time.time() - end)

        # Perform prediction and compute loss wrt objective function
        output = model(imgs)

        # Computes losses and metrics
        loss, metrics = loss_fn(labels, output)

        # Update metrics
        for stat, metric in zip(metrics_values, metrics):
            stat.update(metric, imgs.size(0))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update batch time and progress
        batch_time.update(time.time() - end)
        end = time.time()
        if verbose:
            progress.display(batch_index)

    if verbose:
        progress.display(len(dataset) - 1, last=True)
    # Return average loss and average metrics
    return [m.avg for m in metrics_values]


def evaluate(
        dataset: Iterable,
        model: nn.Module,
        loss_fn: nn.Module,
        device: Optional[str] = 'cpu',
        verbose: Optional[bool] = False,
        prefix: Optional[str] = 'Val',
) -> List[float]:
    """ Evaluate model on a dataset

    :param dataset: Dataset
    :param model: Trained model
    :param loss_fn: Prototype loss and metrics
    :param device: Target device
    :param verbose: Verbose mode
    :param prefix: Prefix for progress bar
    :return: List of metrics
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    metrics_values = [AverageMeter(n, ':6.3f') for n in loss_fn.metrics]
    if verbose:
        progress = ProgressMeter(
            len(dataset),
            [batch_time, data_time] + metrics_values,
            prefix=f"{prefix}: ",
        )

    # Set model validation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_index, (imgs, labels) in enumerate(dataset):
            # Map to device
            imgs = imgs.to(device=device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Data loading time
            data_time.update(time.time() - end)

            # Perform prediction and compute loss wrt objective function
            output = model(imgs)

            # Computes losses and metrics
            loss, metrics = loss_fn(labels, output)

            # Update metrics
            for stat, metric in zip(metrics_values, metrics):
                stat.update(metric, imgs.size(0))

            # Update batch time and progress
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose:
                progress.display(batch_index)
    if verbose:
        progress.display(len(dataset) - 1, last=True)
    # Return average metrics
    return [m.avg for m in metrics_values]


def training_add_parser_options(ap: ArgumentParser) -> None:
    """ Add common training options

    :param ap: Target parser
    """
    ap.add_argument('-m', '--model', required=True, type=str,
                    metavar='<path_to_file>',
                    help='Path to input model.')
    ap.add_argument('-o', '--output', required=True, type=str,
                    metavar='<path_to_file>',
                    help='Path to output model.')
    ap.add_argument('--overwrite', required=False, action='store_true',
                    help='Overwrite output model (if present).')
    ap.add_argument('--epochs', required=True, type=int,
                    metavar='<number_of_epochs>',
                    default=30,
                    help='Number of training epochs. Default: 30')
    ap.add_argument('--device', required=False, type=str,
                    metavar='<name>',
                    default='cpu',
                    help='Target device')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help="Verbose mode")

    # Add options
    dataloader_add_parser_options(ap)
    optimizer_add_parser_options(ap)
    checkpoint_add_parser_options(ap)


def training_init(
        args: Namespace,
) -> Tuple[nn.Module, Tuple[Iterable], Optimizer, object, int, Any]:
    """ Init training from command line

    :param args: Result of argparse
    :return: model, dataloaders, device, optimizer, scheduler, \
            initial epoch, number of epochs, \
            [metrics]
    """
    # Check if cuda is available
    if args.device.startswith('cuda'):
        if not (torch.cuda.is_available()):
            raise ValueError('No GPU available')
        torch.cuda.set_device(args.device)

    # Check output model name
    if Path(args.output).exists() and not args.overwrite:
        raise ValueError(f"Output model {args.output} already exists.")

    # Load model
    model = serialization.load(args.model, map_location=args.device)

    # Split dataset
    dataloaders = dataloaders_from_parser(args)

    # Init optimizer and scheduler
    opt, scheduler = optimizer_from_parser(model, args)

    # Default initialization
    init_epoch = 0
    metrics = None

    # Restart from checkpoint?
    if args.checkpoint_from is not None:
        init_epoch, metrics = load_checkpoint(
            args.checkpoint_from,
            model=model,
            opt=opt,
            device=args.device,
            verbose=args.verbose
        )
    return model, dataloaders, opt, scheduler, init_epoch, metrics


def training_loop(
        model: nn.Module,
        dataloaders: List[Iterable],
        num_epochs: int,
        opt: Optimizer,
        loss_fn: Callable,
        output: str,
        decision_callback: Callable,
        init_epoch: int = 0,
        scheduler: object = None,
        loading_callback: Callable = None,
        checkpoint_file: str = None,
        checkpoint_freq: int = 0,
        device: str = 'cpu',
        verbose: bool = False,
) -> None:
    """ Main training loop

    :param model: Source model
    :param dataloaders: Train/test/validation dataloaders
    :param num_epochs: Number of epochs
    :param opt: Optimizer
    :param loss_fn: Loss function
    :param output: Path to output file
    :param decision_callback: Callback using the epoch metrics to take a decision
    :param init_epoch: Index of first epoch
    :param scheduler: Learning rate scheduler
    :param loading_callback: Callback used to load final optimized model
    :param checkpoint_file: Path to checkpoint file (if any)
    :param checkpoint_freq: Checkpoint frequency
    :param device: Target device
    :param verbose: Verbose mode
    """
    [trainset, testset, valset] = dataloaders

    # Main loop
    for epoch in range(init_epoch, num_epochs):
        # Train for one epoch
        epoch_metrics = train(
            dataset=trainset,
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            epoch=epoch,
            device=device,
            verbose=verbose,
        )

        # Evaluate on validation set (overwrite epoch metrics)
        if valset:
            epoch_metrics = evaluate(
                dataset=valset,
                model=model,
                loss_fn=loss_fn,
                device=device,
                verbose=verbose,
                prefix='Valid',
            )

        # Take a decision based on the metrics
        best_metrics = decision_callback(model, output, epoch_metrics)

        # Save checkpoint?
        save_checkpoint(
            filename=checkpoint_file,
            model=model,
            metrics=best_metrics,
            epoch=epoch,
            opt=opt,
            freq=checkpoint_freq,
            verbose=verbose,
        )

        # Learning rate scheduler
        if scheduler:
            if scheduler.__class__.__name__ in ['StepLR']:
                scheduler.step()
            elif scheduler.__class__.__name__ in ['ReduceLROnPlateau']:
                scheduler.step(epoch_metrics[0])

    # Early stop
    if not verbose or testset is None:
        return

    # Load best model
    if loading_callback is None:
        model = serialization.load(output, map_location=device)
    else:
        model = loading_callback(output, device)

    # Evaluate model on test set
    test_metrics = evaluate(
        dataset=testset,
        model=model,
        loss_fn=loss_fn,
        device=device,
        verbose=True,
        prefix='Test'
    )
    for n, value in zip(loss_fn.metrics, test_metrics):
        print(f'Test avg {n}: {value}')


def evaluation_add_parser_options(ap: ArgumentParser) -> None:
    """ Add common evaluation options

    :param ap: Target parser
    """
    ap.add_argument('-m', '--model', required=True, type=str,
                    metavar='<path_to_file>',
                    help='Path to model.')
    ap.add_argument('--only', required=False, type=str,
                    metavar='<train/test/valid>',
                    choices=['train', 'test', 'valid'],
                    help='Evaluate on subset only')
    ap.add_argument('--device', required=False, type=str,
                    metavar='<name>',
                    default='cpu',
                    help='Target device')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help="Verbose mode")

    # Add dataset options
    dataloader_add_parser_options(ap)


def evaluation_init(
        args: Namespace,
) -> Tuple[nn.Module, List[Iterable]]:
    """ Init evaluation from command line

    :param args: Result of argparse
    :return: model, dataloaders
    """
    # Check if cuda is available
    if args.device.startswith('cuda'):
        if not (torch.cuda.is_available()):
            raise ValueError('No GPU available')
        torch.cuda.set_device(args.device)

    # Load model
    model = serialization.load(args.model, map_location=args.device)

    # Split dataset
    dataloaders = dataloaders_from_parser(args)
    # Select only one if necessary
    dataloaders = [dataloader if (not args.only or args.only == name) else None
                   for dataloader, name in zip(dataloaders, ['train', 'test', 'valid'])]

    return model, dataloaders


def evaluation_loop(
        model: nn.Module,
        dataloaders: List[Iterable],
        loss_fn: Callable,
        device: str,
        verbose: bool,
) -> None:
    """ Evaluate model on datasets

    :param model: Source model
    :param dataloaders: Train/test/validation dataloaders
    :param loss_fn: Loss function
    :param device: Target device
    :param verbose: Verbose mode
    """
    [trainset, testset, valset] = dataloaders

    for dataset, name in zip([trainset, testset, valset], ['Train', 'Test', 'Val']):
        if dataset:
            metrics = evaluate(dataset, model, loss_fn, device, verbose, name)
            if verbose:
                for n, value in zip(loss_fn.metrics, metrics):
                    print(f'{name} avg {n}: {value}')
