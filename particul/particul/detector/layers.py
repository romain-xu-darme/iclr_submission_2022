import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class ChannelWiseSoftmax(nn.Module):
    """ Channel-wise Softmax

    When given a tensor of shape ``N x C x H x W``, apply Softmax to each channel
    """

    def forward(self, input: Tensor) -> Tensor:
        """ Forward layer function

        :param input: Input 4D tensor (expected shape ``N x C x H x W``)
        :return: 4D tensor (same shape as input) obtained after applying softmax on each
            of the ``C`` channels.
        """
        assert input.dim() == 4, 'ChannelWiseSoftmax requires a 4D tensor as input'
        # Reshape tensor to N x C x (H x W)
        reshaped = input.view(*input.size()[:2], -1)
        # Apply softmax along 2nd dimension than reshape to original
        return nn.Softmax(2)(reshaped).view_as(input)


class ConfidenceMeasure(nn.Module):
    """ This layer computes a confidence measure as the cumulative distribution function
    of a logistic distribution which is calibrated over the training set.
    """

    def __init__(self) -> None:
        super(ConfidenceMeasure, self).__init__()
        self._mean = nn.Parameter(torch.FloatTensor([-1.0 * float('inf')]),
                                  requires_grad=False)
        self._std = nn.Parameter(torch.FloatTensor([1.0]),
                                 requires_grad=False)

    @property
    def calibrated(self) -> bool:
        """ Layer is calibrated when mean value is not -infinity """
        return self._mean.item() != -1.0 * float('inf')

    def calibrate(self, mean: float, std: float) -> None:
        """ Calibrate measure

        :param mean: Logistic distribution mean value
        :param std: Logistic distribution standard deviation
        """
        self._mean = nn.Parameter(torch.FloatTensor([mean]), requires_grad=False)
        self._std = nn.Parameter(torch.FloatTensor([std]), requires_grad=False)

    def forward(self, input: Tensor) -> Tensor:
        """ Forward pass. Compute confidence measure as sigmoid((max(x)-mean)/std)

        :param input: Input tensor (shape ``N x 1 x H x W``)
        :return: Output tensor (shape ``N x 1``)
        """
        vmax = nn.AdaptiveMaxPool2d(1)(input)[..., 0, 0]
        return torch.sigmoid(vmax.sub_(self._mean).div_(self._std))

    def extra_repr(self) -> str:
        res = f'calibrated: {self.calibrated}'
        if self.calibrated:
            res += f', mean: {self._mean.item()}, std: {self._std.item()}'
        return res


class PatternDetector(nn.Module):
    """ Pattern detector built as 1x1 convolution with optional activation
    followed a normalization layer.
    """

    def __init__(self,
                 nchannels: int,
                 activation: Optional[str] = None,
                 ) -> None:
        """

        :param nchannels: Number of input channels
        :param activation: Activation function
        """
        super(PatternDetector, self).__init__()

        # Build convolutional layer and filter
        self.conv = nn.Conv2d(
            in_channels=nchannels,
            out_channels=1,
            kernel_size=(1, 1),
            padding='same',
            bias=False,
        )
        # Glorot initialization
        nn.init.xavier_normal_(self.conv.weight)
        activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}
        self.act = activations[activation] if activation else None
        self.confidence = ConfidenceMeasure()
        self.norm = ChannelWiseSoftmax()
        # By default, normalization is enabled
        self.enable_normalization = True

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass

        Apply convolution (with optional activation function), followed by filter layer.
        Finally, apply ChannelWiseSoftmax only if normalization is enabled.
        """
        x = self.conv(x)
        if self.act:
            x = self.act(x)
        c = self.confidence(x)
        if self.enable_normalization:
            x = self.norm(x)
        return x, c

    def calibrate(self, **kwargs):
        """ Calibrate confidence measure
        """
        self.confidence.calibrate(**kwargs)

    @property
    def calibrated(self) -> bool:
        return self.confidence.calibrated

    def train(self, mode: bool = True) -> nn.Module:
        """ Overwrite train() function.

        :param mode: Train (true) or eval (false)
        """
        self.conv.weight.requires_grad = mode
        return self
