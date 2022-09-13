import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List
from typing import Optional


class FocalLoss(nn.Module):
    """ Focal loss for categorical cross entropy """

    def __init__(self, gamma: Optional[float] = 0) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.metrics = ['loss', 'accuracy']

    def forward(self, labels: Tensor, logits: Tensor) -> Tuple[Tensor, List[float]]:
        """ Compute focal loss and accuracy

        :param labels: Label tensor of shape N (x 1)
        :param logits: Output logits of shape N x nclasses
        :return: Loss tensor, [accuracy]
        """
        # Focal loss
        pt = logits.softmax(dim=1).gather(1, labels.view(-1, 1))
        logpt = pt.log()
        fl = -1 * (1 - pt) ** self.gamma * logpt

        # Accuracy
        if labels.dim() > 1:
            labels = labels.squeeze(dim=1)
        pindex = torch.argmax(logits, dim=1)
        npos = torch.sum(pindex == labels).item()
        loss = fl.mean()
        return loss, [loss.item(), npos / logits.size(0)]


class ClassificationLoss(FocalLoss):
    """ Classification loss (categorical cross-entropy) """

    def __init__(self) -> None:
        super(ClassificationLoss, self).__init__(gamma=0)
