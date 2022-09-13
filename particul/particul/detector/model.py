import torch
import torch.nn as nn
import torchvision.models as torch_models
from torch import Tensor
from typing import Optional, Union, Tuple, Dict
from argparse import ArgumentParser, Namespace
from .layers import PatternDetector

# Handles optional packages
try:
    from tf2torch.json2torch import ModelFromJson
except ModuleNotFoundError:
    pass

# List of supported pre-trained backbones
bb_supported = [mname for mname in torch_models.__dict__
                if mname.islower() and not mname.startswith("__")
                and callable(torch_models.__dict__[mname])]


class Particul(nn.Module):
    """ Particul: Part Identification through fast Converging Unsupervised Learning """

    def __init__(self,
                 backbone: str,
                 npatterns: int,
                 activation: Optional[str] = None,
                 ) -> None:
        """
        :param backbone: Backbone of the CNN feature extractor
        :param npatterns: Number of learnable patterns
        :param activation: Detector activation function

        'backbone' can be either the name of a supported network architecture or the
        path to an existing .pth file.
        """
        super(Particul, self).__init__()

        # Check arguments
        if npatterns <= 0:
            raise ValueError(f"[Particul] Invalid configuration: npatterns={npatterns}")
        self.npatterns = npatterns

        # Recover backbone model and extract last convolutional layer
        if backbone in bb_supported:
            self.backbone = self._bb_trim(torch_models.__dict__[backbone](pretrained=True))
        else:
            backbone = torch.load(backbone, map_location='cpu')
            self.backbone = self._bb_trim(backbone)
            if self.backbone is None:
                print('Warning: Could not find pooling layer in backbone, '
                      'keeping entire model')
                self.backbone = backbone

        # Recover number of backbone feature channels
        self.backbone.eval()
        self.nfchans = self.backbone(torch.rand(1, 3, 224, 224)).data.size(1)

        # Part detectors
        self.detectors = nn.ModuleList(
            [PatternDetector(
                nchannels=self.nfchans,
                activation=activation,
            ) for _ in range(self.npatterns)]
        )

        # Freeze backbone by default
        self.trainable_backbone = False
        # By default, detectors are trainable
        self.trainable_detectors = True

        # By default, do not output backbone features
        self.enable_features = False
        # By default, do not output confidence measure
        self.enable_confidence = False

    def forward(self,
                x: Tensor
                ) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """ Forward pass.

        Apply backbone feature extraction. Compute activation maps out of each
        pattern detector, then concatenate all activation maps.
        If feature output is enabled, also output backbone features
        """
        f = self.backbone(x)
        # Aggregate results from all detectors
        x, c = zip(*[d(f) for d in self.detectors])
        res = torch.cat(x, dim=1)

        # Activation maps only
        if not self.enable_confidence and not self.enable_features:
            return res

        # Output is a tuple
        res = (res,)
        if self.enable_confidence:
            res += (torch.cat(c, dim=1),)
        if self.enable_features:
            res += (f,)
        return res

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

    def train(self, mode: bool = True) -> nn.Module:
        """ Overwrite train() function to freeze elements if necessary

        :param mode: Train (true) or eval (false)
        """
        self.backbone.train(mode and self.trainable_backbone)
        for param in self.backbone.parameters():
            param.requires_grad = mode and self.trainable_backbone
        for d in self.detectors:
            d.train(mode and self.trainable_detectors)
        return self

    @staticmethod
    def compare_state_dicts(ref: Dict, tst: Dict) -> bool:
        """ Compare two state dictionnaries

        :param ref: Reference dictionnary
        :param tst: Test dictionnary
        :return: True if and only if dictionnaries are identical
        """
        if len(ref) != len(tst):
            print(f'Mismatching lengths of state dicts {len(ref)} v. {len(tst)}')
            return False

        for ref_entry, tst_entry in zip(ref.items(), tst.items()):
            ref_key, ref_val = ref_entry[0], ref_entry[1]
            tst_key, tst_val = tst_entry[0], tst_entry[1]
            if ref_key != tst_key:
                print(f'Mismatching state dict keys {ref_key} v. {tst_key}')
                return False
            if isinstance(ref_val, str):
                print(ref_val)
            if not torch.equal(ref_val, tst_val):
                print(f'Mismatching state dict values for {ref_key}')
                return False
        return True

    def compare(self, model: nn.Module, mode: str) -> bool:
        """ Compare models

        :param model: Model to be compared with
        :param mode: Comparison mode encoded as a combination of letters

        - b: Same backbone
        - d: Same detectors

        :return: True if and only if models are identical w.r.t. mode
        """
        if self.__class__.__name__ != model.__class__.__name__:
            print(f'Mismatching module types {self.__class__.__name__} v. \
                    {model.__class__.__name__}')
        res = True
        if 'b' in mode:
            # Check that backbones are identical
            res = res and Particul.compare_state_dicts(self.backbone.state_dict(),
                                                       model.backbone.state_dict())
        if 'd' in mode:
            if self.npatterns != model.npatterns:
                print(f'Mismatching number of patterns {self.npatterns} v. \
                    {model.npatterns}')
            for ref_d, tst_d in zip(self.detectors, model.detectors):
                res = res and Particul.compare_state_dicts(ref_d.state_dict(),
                                                           tst_d.state_dict())
        return res

    @staticmethod
    def _bb_trim(model: nn.Module) -> nn.Module:
        """ Find the last average pooling layer in a model
        and remove all subsequent layers

        :param model: Target model
        :return: Trimmed model
        """
        children = list(model.children())
        children.reverse()
        while children:
            child = children.pop(0)
            if child.__class__.__name__ == 'AdaptiveAvgPool2d':
                break
            if child.__class__.__name__ == 'Sequential':
                # Recursive call
                submodel = Particul._bb_trim(child)
                if submodel is not None:
                    # Average pooling was found in submodel
                    children.insert(0, submodel)
                    break
        if children:
            children.reverse()
            return nn.Sequential(*children)
        return None

    @staticmethod
    def add_parser_options(parser: ArgumentParser) -> None:
        """ Add all options for the initialization of a Particul model
        """
        group = parser.add_argument_group("Particul")
        group.add_argument('--particul-backbone', type=str, required=True,
                           metavar='<name or path_to_model>',
                           help='Name of pretrained model used as backbone for the architecture.')
        group.add_argument('--particul-npatterns', type=int, required=True,
                           metavar='<num>',
                           help='Number of pattern detectors.')
        group.add_argument('--particul-activation', type=str, required=False,
                           metavar='<func>',
                           help='Activation function for pattern detectors.')

    @staticmethod
    def build_from_parser(args: Namespace) -> nn.Module:
        """ Build Particul object from argparse arguments
        """
        return Particul(
            backbone=args.particul_backbone,
            npatterns=args.particul_npatterns,
            activation=args.particul_activation,
        )
