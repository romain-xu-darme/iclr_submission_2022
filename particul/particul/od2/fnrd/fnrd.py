from tqdm import tqdm
import torch
import torch_toolbox
from torch_toolbox.datasets.dataset import dataloader_add_parser_options, dataloaders_from_parser
import numpy as np
from numpy import ndarray
from torch import Tensor
from typing import Tuple, Optional
import torch.nn as nn
from torch.utils.data import DataLoader
from particul.od2.fnrd.resnet import resnet50probed
from particul.od2.fnrd.vgg import which_vgg, _vgg
from argparse import ArgumentParser, Namespace


class FNRDModel(nn.Module):

    def __init__(self, classifier: nn.Module, bounds: Optional[ndarray] = None) -> None:
        """ FNRDModel is the combination of a probed classifier and a set of min-max activation bounds
        corresponding to the Maximum Function Region of the classifier over a given dataset

        :param classifier: Target classifier (supported: ResNet50)
        :param bounds: Min-Max bounds
        """
        super(FNRDModel, self).__init__()

        self.classifier = None
        if classifier.__class__.__name__ == "ResNet":
            nclasses = classifier.fc.out_features
            self.classifier = resnet50probed(nclasses)
            self.classifier.load_state_dict(classifier.state_dict())
        elif classifier.__class__.__name__ == "VGG":
            nclasses, cfg, batch_norm = which_vgg(classifier)
            self.classifier = _vgg(cfg=cfg,batch_norm=batch_norm,weights=None,progress=False,num_classes=nclasses)
            self.classifier.load_state_dict(classifier.state_dict())
        else:
            raise ValueError("Architecture not supported")

        self.min_mask = None
        self.max_mask = None
        if bounds is not None:
            self.min_mask = torch.from_numpy(bounds[0])
            self.max_mask = torch.from_numpy(bounds[1])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """ Forward pass of the classifier with dynamic computation of the FNRD confidence score (1-FNRD)

        :param x: Input tensor
        :returns: logits, scores
        """
        if self.min_mask is None:
            raise ValueError('MFR not set.')
        logits = self.classifier(x)
        activations = self.classifier.get_activations()
        max_outliers = torch.sum(activations > self.max_mask, axis=1)  # Shape N
        min_outliers = torch.sum(activations < self.min_mask, axis=1)  # Shape N
        conf = 1 - (max_outliers + min_outliers) / len(self.min_mask)
        return logits, conf

    def to(self, device: str) -> nn.Module:
        """ Overrides to() method.

        :param device: Target
        :returns: self
        """
        self.classifier = self.classifier.to(device)
        if self.min_mask is not None:
            self.min_mask = self.min_mask.to(device)
            self.max_mask = self.max_mask.to(device)
        return self

    def train(self, mode: bool = True) -> nn.Module:
        """ Overrides train() method.

        :param mode: Training mode
        :returns: self
        """
        self.classifier.train(mode)
        return self

    def compute_mfr(self, dataloader: DataLoader, device: str, verbose: bool) -> Tuple[Tensor, Tensor]:
        """ Compute the Maximum Function Region (MFR) of a model. The MFR of a neuron is the min and max
        post-activation values for the whole training dataset.

        :param dataloader: Training set
        :param device: Target device
        :param verbose: Verbose mode
        """
        dataloader = tqdm(dataloader) if verbose else dataloader
        min_mask, max_mask = None, None
        for batch_idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            self.classifier(imgs)
            activations = self.classifier.get_activations()
            min_mask = torch.minimum(activations, min_mask)[0] if batch_idx > 0 else activations.min(dim=0)[0]
            max_mask = torch.maximum(activations, max_mask)[0] if batch_idx > 0 else activations.max(dim=0)[0]

        return min_mask, max_mask

    @staticmethod
    def add_parser_options(parser: ArgumentParser, add_mfr: Optional[bool] = False) -> None:
        """ Add all options for the initialization of a FNRDModel

        :param parser: Target argparse parser
        :param add_mfr: Add options to compute the model's MFR
        """
        parser.add_argument('--fnrd-classifier', type=str, required=True,
                            metavar='<path_to_model>',
                            help='Path to pretrained classifier.')
        parser.add_argument('--fnrd-mfr', type=str, required=True,
                            metavar='<path_to_file>',
                            help='Path to NPY file containing MFR.')
        if add_mfr:
            dataloader_add_parser_options(parser)
            parser.add_argument('--device', type=str, required=False,
                                metavar='<device_id>',
                                default='cpu',
                                help='Target device for execution.')
            parser.add_argument('-v', "--verbose", action='store_true',
                                help="Verbose mode")

    @staticmethod
    def build_from_parser(args: Namespace) -> nn.Module:
        """ Build FNRDModel from argparse arguments

        :param args: Command line argument
        """
        return FNRDModel(classifier=torch_toolbox.load(args.fnrd_classifier),
                         bounds=np.load(args.fnrd_mfr)).to(args.device)

    @staticmethod
    def compute_mfr_from_parser(args: Namespace) -> None:
        """ Compute MFR from argparse arguments

        :param args: Command line argument
        """
        # Load trainset set only
        trainloader, _, _ = dataloaders_from_parser(args)
        # Load model
        model = FNRDModel(
            classifier=torch_toolbox.load(args.fnrd_classifier),
            bounds=None).to(args.device)
        model.eval()
        min_bounds, max_bounds = model.compute_mfr(trainloader, args.device, args.verbose)
        mfr = np.array([bound.detach().cpu().numpy() for bound in [min_bounds, max_bounds]])
        np.save(args.fnrd_mfr, mfr)
