import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List, Optional, Union
from argparse import ArgumentParser, Namespace


class LocalityLoss(nn.Module):
    def __init__(self, npatterns: int, ksize: Optional[int] = 3) -> None:
        """ Locality loss

        Purpose: Maximize activation is a single location.

        :param npatterns: Number of pattern detectors
        :param ksize: Uniform kernel size
        """
        super(LocalityLoss, self).__init__()
        # Init uniform kernel
        self.conv = nn.Conv2d(
            in_channels=npatterns,
            out_channels=npatterns,
            kernel_size=ksize,
            padding='same',
            groups=npatterns,
            bias=False
        )
        self.conv.weight = nn.Parameter(torch.ones(self.conv.weight.size()))

    def forward(self, output: Tensor) -> Tensor:
        """ Compute locality loss.

        L(output) = -max(output*u), where u is a 3x3 uniform kernel used to relax the
        locality constraint.
        """
        # Convolution with uniform kernel
        F = self.conv(output)
        # Maxpool
        vmax = nn.AdaptiveMaxPool2d(1)(F).view(F.size()[:2])
        # Return average loss PER DETECTOR
        return -torch.mean(vmax, dim=0)


class UnicityLoss(nn.Module):
    def __init__(self, threshold: float) -> None:
        """ Unicity loss

        Purpose: Ensure the diversity of activation maps across detectors

        :param threshold: Unicity threshold
        """
        super(UnicityLoss, self).__init__()
        self.threshold = threshold

    def forward(self, output: Tensor) -> Tensor:
        """ Compute unicity loss.

        L(output) = ReLU(max(S)-t), where S is the sum of all activation maps
        """
        # Compute sum of all part activations
        S = torch.sum(output, 1)
        # Maxpool
        vmax = nn.AdaptiveMaxPool2d(1)(S).view(S.size()[:1])
        return torch.mean(torch.relu(vmax - self.threshold))


class ClusteringLoss(nn.Module):
    def __init__(self, threshold: float) -> None:
        """ Clustering loss

        Purpose: Ensure the clustering of activation maps across detectors

        :param threshold: Clustering threshold
        """
        super(ClusteringLoss, self).__init__()
        self.threshold = threshold
        # Init edge kernel
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding='same',
            bias=False
        )
        kernel = -torch.ones(self.conv.weight.size())
        kernel[0, 0, 1, 1] = 1
        self.conv.weight = nn.Parameter(kernel)

    def forward(self, output: Tensor) -> Tensor:
        """ Compute clustering loss.

        L(output) = ReLU(max(S*e)-t), where S is the sum of all activation maps
        """
        # Compute sum of all part activations
        S = torch.sum(output, 1, keepdim=True)
        # Convolution with edge kernel
        F = self.conv(S)
        # Maxpool
        vmax = nn.AdaptiveMaxPool2d(1)(F).view(F.size()[:2])
        return torch.mean(torch.relu(vmax - self.threshold))


class ParticulLoss(nn.Module):
    def __init__(self,
                 npatterns: int,
                 loc_ksize: int,
                 unq_ratio: float,
                 unq_thres: float,
                 cls_ratio: float,
                 cls_thres: float
                 ) -> None:
        """ Particul loss: weighted sum of Locality, Unicity and Clustering losses

        :param npatterns: Number of patterns
        :param loc_ksize; Locality uniform kernel size
        :param unq_ratio: Unicity ratio
        :param unq_thres: Unicity threshold
        :param cls_ratio: Clustering ratio
        :param cls_thres: Clustering threshold
        """
        super(ParticulLoss, self).__init__()
        self.unq_ratio = unq_ratio
        self.cls_ratio = cls_ratio
        self.locality_loss = LocalityLoss(npatterns, loc_ksize)
        self.unicity_loss = UnicityLoss(unq_thres)
        self.clustering_loss = ClusteringLoss(cls_thres)
        self.metrics = ['Detection', 'Locality', 'Unicity', 'Clustering']
        # Add locality loss per detector
        self.metrics += [f'Loc {p}' for p in range(npatterns)]

    def forward(self, labels: Tensor, output: Union[Tensor, Tuple]) -> Tuple[Tensor, List]:
        """ Compute Particul loss

        :param labels: Ignored (unsupervised setting)
        :param output: Tensor of pattern activation maps or tuple of Tensors
        """
        if isinstance(output, tuple):
            output = output[0]
        l_loss = self.locality_loss(output)  # Shape: [npatterns]
        u_loss = self.unicity_loss(output)
        c_loss = self.clustering_loss(output)
        loss = torch.mean(l_loss) + self.unq_ratio * u_loss + self.cls_ratio * c_loss
        metrics = [loss.item(), torch.mean(l_loss).item(), u_loss.item(), c_loss.item()]
        metrics += [l.item() for l in l_loss]
        return loss, metrics

    @staticmethod
    def add_parser_options(parser: ArgumentParser):
        """ Add all options for the initialization of a Particul loss function
        """
        parser.add_argument('--particul-loss-locality-ksize', type=int, required=False,
                            metavar='<kernel_size>',
                            default=3,
                            help='Size of uniform kernel in "Locality" part loss (default: 3)')
        parser.add_argument('--particul-loss-unicity', type=float, required=False,
                            metavar='<ratio>',
                            default=1.0,
                            help='Ratio for the "Unicity" part loss.')
        parser.add_argument('--particul-loss-clustering', type=float, required=False,
                            metavar='<ratio>',
                            default=0.0,
                            help='Ratio for the "Clustering" part loss')

    @staticmethod
    def build_from_parser(npatterns: int, args: Namespace):
        """ Build Particul loss function from argparse arguments

        :param npatterns: Number of patterns
        :param args: Parser output
        """
        loc_ksize = args.particul_loss_locality_ksize
        unq_ratio = args.particul_loss_unicity
        unq_thres = 1.0
        cls_ratio = args.particul_loss_clustering
        cls_thres = 0.0

        return ParticulLoss(npatterns, loc_ksize, unq_ratio, unq_thres, cls_ratio, cls_thres)
