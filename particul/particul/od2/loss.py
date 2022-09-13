import torch
import torch.nn as nn
from particul.detector.loss import ParticulLoss
from typing import Tuple, List
from torch import Tensor
from argparse import Namespace


class ParticulOD2Loss(ParticulLoss):
    def __init__(self, *args, **kwargs) -> None:
        super(ParticulOD2Loss, self).__init__(*args, **kwargs)
        # Remove individual locality losses from metrics
        self.metrics = ['Detection', 'Locality', 'Unicity', 'Clustering']

    def forward(self, labels: Tensor, output: Tuple[Tensor, Tensor]) -> Tuple[Tensor, List]:
        """ Compute ParticulOD2 loss

        :param labels: Batch labels (shape N x 1)
        :param output: Classifier output (N x C), Activation maps (N x C x P x H x W)
        :returns: loss, metrics
        """
        # TODO: When labels is None, use classifier prediction instead!
        amaps = output[1]
        # Expand labels to one-hot of shape N x C x 1 x 1 x 1
        nclasses = amaps.size(1)
        if labels.dim() > 1:
            labels = labels.squeeze(dim=1)
        labels = nn.functional.one_hot(labels, num_classes=nclasses)
        labels = labels[..., None, None, None]
        # Recover activation maps corresponding to labels
        # indexed has shape N x P x H x W
        indexed = torch.sum(amaps * labels, dim=1)
        l_loss = torch.mean(self.locality_loss(indexed))
        u_loss = self.unicity_loss(indexed)
        c_loss = self.clustering_loss(indexed)
        loss = l_loss + self.unq_ratio * u_loss + self.cls_ratio * c_loss
        metrics = [loss.item(), l_loss.item(), u_loss.item(), c_loss.item()]
        return loss, metrics

    @staticmethod
    def build_from_parser(npatterns: int, args: Namespace):
        """ Overrides Particul function

        :param npatterns: Number of patterns
        :param args: Parser output
        """
        return ParticulOD2Loss(
            npatterns=npatterns,
            loc_ksize=args.particul_loss_locality_ksize,
            unq_ratio=args.particul_loss_unicity,
            unq_thres=1.0,
            cls_ratio=args.particul_loss_clustering,
            cls_thres=0.0,
        )
