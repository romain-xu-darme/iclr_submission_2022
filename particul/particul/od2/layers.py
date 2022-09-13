import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from particul.detector.layers import PatternDetector


class ClassWiseParticul(nn.Module):
    """ Particul module trained only on a given class """

    def __init__(self,
                 nchannels: int,
                 npatterns: int,
                 activation: Optional[str] = None,
                 ) -> None:
        """ Initialize ClassWiseParticul block

        :param nchannels: Number of feature channels
        :param npatterns: Number of patterns to learn
        :param activation: Activation function
        """
        super(ClassWiseParticul, self).__init__()
        self.detectors = nn.ModuleList([
            PatternDetector(nchannels, activation) for _ in range(npatterns)])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """ Forward pass

        :param x: Input tensor of shape N x D x H x W
        :return: Activation maps (N x P x H x W), Confidence measures (N x P)
        """
        # Aggregate results from all detectors
        x, c = zip(*[d(x) for d in self.detectors])
        x = torch.cat(x, dim=1)
        c = torch.cat(c, dim=1)
        return x, c

    @property
    def enable_normalization(self) -> bool:
        """ True if and only if normalization is enabled on all detectors
        """
        return all([d.enable_normalization for d in self.detectors])

    @enable_normalization.setter
    def enable_normalization(self, val: bool) -> None:
        """ Enable/disable normalization layer on each detector
        """
        for d in self.detectors:
            d.enable_normalization = val

    @property
    def calibrated(self) -> bool:
        """ Return true if and only if all detectors have been calibrated
        """
        return all([d.calibrated for d in self.detectors])

    def calibrate(self, index: int, **kwargs) -> None:
        """ Set calibration values on a given detector

        :param index: Detector index
        :param kwargs: Calibration values
        """
        self.detectors[index].calibrate(**kwargs)
