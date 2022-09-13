import torch
import torch.nn as nn
import torch_toolbox
from .layers import ClassWiseParticul
from torch.nn.modules.module import _addindent
from torch import Tensor
from typing import Tuple, Optional
from argparse import ArgumentParser, Namespace


class ParticulOD2(nn.Module):
    """ Out-of distribution detection (OD2) using Particul detectors """

    def __init__(self,
                 nclasses: int,
                 npatterns: int,
                 nchans: int,
                 source: str,
                 ) -> None:
        """
        :param nclasses: Number of classes
        :param npatterns: Number of patterns per class
        :param nchans: Number of feature channels extracted from the source model
        :param source: Name of source layer
        """
        super(ParticulOD2, self).__init__()
        self.npatterns = npatterns
        self.nclasses = nclasses
        self.nchans = nchans
        self.source = source

        # Init Class-wise blocks
        self.particuls = nn.ModuleList([ClassWiseParticul(self.nchans, self.npatterns)
                                        for _ in range(self.nclasses)])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """ Forward pass

        :param x: Tensor (N x D x H x W)
        :return: Activation maps (N x C x P x H x W), [confidence (N x C x P)]
        """
        # Aggregate results from all class-wise detectors
        x, c = zip(*[p(x) for p in self.particuls])
        x = torch.stack(x, dim=1)
        c = torch.stack(c, dim=1)
        return x, c

    @property
    def calibrated(self) -> bool:
        """ Return true if and only if all detectors have been calibrated
        """
        return all([particul.calibrated for particul in self.particuls])

    @property
    def enable_normalization(self) -> bool:
        """ True if and only if normalization is enabled on all detectors
        """
        return all([p.enable_normalization for p in self.particuls])

    @enable_normalization.setter
    def enable_normalization(self, val: bool) -> None:
        """ Enable/disable normalization layer on each detector
        """
        for p in self.particuls:
            p.enable_normalization = val

    def __repr__(self):
        """ Overwrite __repr__ to display a single detector """
        main_str = self._get_name() + ' [connected to ' + self.source + '] ('
        child_lines = [f'(particuls): {self.nclasses} x ' +
                       _addindent(repr(self.particuls[0]), 2)]
        main_str += '\n  ' + '\n  '.join(child_lines) + '\n)'
        return main_str

    def save(self, path: str) -> None:
        """ Save model

        :param path: Path to destination file
        """
        torch.save({
            'module': 'particul.od2.model',
            'class': 'ParticulOD2',
            'loader': 'load',
            'nclasses': self.nclasses,
            'npatterns': self.npatterns,
            'nchans': self.nchans,
            'source': self.source,
            'state_dict': self.state_dict(),
        }, path)

    @staticmethod
    def load(path: str, map_location='cpu') -> nn.Module:
        """ Load model

        :param path: Path to source file
        :param map_location: Target device
        """
        infos = torch.load(path, map_location='cpu')
        model = ParticulOD2(
            infos['nclasses'],
            infos['npatterns'],
            infos['nchans'],
            infos['source'],
        )
        model.load_state_dict(infos['state_dict'])
        return model.to(map_location)

    @staticmethod
    def add_parser_options(parser: ArgumentParser) -> None:
        """ Add all options for the initialization of a ParticulOD2 model

        :param parser: Target argparse parser
        """
        group = parser.add_argument_group("Out-of-distribution detection")
        group.add_argument('--od2-classifier', type=str, required=True,
                           metavar='<path_to_model>',
                           help='Path to pretrained classifier.')
        group.add_argument('--od2-layer', type=str, required=True,
                           metavar='<name>',
                           help='Name of feature extraction layer.')
        group.add_argument('--od2-npatterns', type=int, required=True,
                           metavar='<num>',
                           help='Number of patterns per class.')
        group.add_argument('--od2-ishape', type=int, required=False, nargs='+',
                           metavar='<shape>',
                           default=[3, 224, 224],
                           help='Classifier input shape (default: [3,224,224]).')
        group.add_argument('--od2-no-class', required=False, action='store_true',
                           help='Ignore classes (used for fine-grained datasets.')
        group.add_argument('--od2-from-particul', type=str, required=False,
                           metavar='<path_to_file>',
                           help='Path to source detector.')

    @staticmethod
    def build_from_parser(args: Namespace) -> nn.Module:
        """ Build ParticulOD2 model from argparse arguments

        :param args: Command line argument
        """
        if args.od2_npatterns < 1:
            raise ValueError(f'[ParticulOD2] Invalid npatterns={args.od2_npatterns}')

        # Extract parameters
        classifier = torch.load(args.od2_classifier, map_location='cpu')
        layer = args.od2_layer
        ishape = args.od2_ishape
        npatterns = args.od2_npatterns

        # Check layer existence
        if not (hasattr(classifier, layer)):
            raise ValueError(f'Unknown layer {layer}')

        # Find number of classes and feature channels
        global nchans
        nchans = None

        def get_nchans():
            def hook(model, input, output):
                global nchans
                nchans = output.size(1)

            return hook

        classifier.eval()
        getattr(classifier, layer).register_forward_hook(get_nchans())
        nclasses = classifier(torch.zeros([1] + ishape)).data.size(1)  # Find number of classes and channels
        # Overwrite number of classes if necessary
        nclasses = 1 if args.od2_no_class else nclasses
        model = ParticulOD2(nclasses, npatterns, nchans, layer)

        if args.od2_from_particul is not None:
            if not(args.od2_no_class):
                raise ValueError('Import from Particul detector only available with --od2-no-class option.')
            src = torch_toolbox.load(args.od2_from_particul)
            for ipattern in range(npatterns):
                model.particuls[0].detectors[ipattern].load_state_dict(src.detectors[ipattern].state_dict())
        return model


class ClassifierWithParticulOD2(nn.Module):
    """ Meta-model including Classifier + Out-of-distribution detection """

    def __init__(self,
                 classifier: nn.Module,
                 detector: nn.Module,
                 ) -> None:
        """ Initialize meta-model

        :param classifier: Path to trained classifier
        :param detector: ParticulOD2 detector
        """
        super(ClassifierWithParticulOD2, self).__init__()

        # Freeze pretrained classifier
        self.classifier = classifier
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False

        # Out-of-distribution detection
        self.detector = detector
        flayer = detector.source

        # Register forward hook in flayer
        # During inference on self.classifier, output of self.classifier.flayer
        # will now be available in self.features
        self.features = None
        getattr(self.classifier, flayer).register_forward_hook(self._hook())
        # Default forward mode
        self._mode = 'logits+amaps'

    def _hook(self):
        """ Internal hook updating the value of self.features """

        def hook(model, input, output):
            self.features = output

        return hook

    @property
    def mode(self) -> str:
        """ Current operation mode

        - In ``logits+amaps`` mode, the model outputs the classifier logits and all activation maps for all classes. \
        This mode is used to train and calibrate the detectors
        - In ``logits+indiv_scores`` mode, the model outputs the classifier logits and the associated confidence \
        scores of each detector. This mode is used to evaluate the quality of individual detectors.
        - In ``production`` mode, the model outputs the classifier logits and a global confidence score computed as \
        average value of each individual detector. This mode is used in production.
        - In ``amaps+indiv_scores`` mode, the model behaves as a Particul detector, dynamcially selecting the \
        activation map and individual confidence scores of the most probable class
        - In ``amaps_only`` mode, the model only outputs the activation map of the most probable class. This mode is \
        used to compute pattern visualizations.
        """
        return self._mode

    @mode.setter
    def mode(self, val: str) -> None:
        """ Set operation mode

        :param val: Operation mode
        """
        if val not in ['production', 'logits+amaps', 'logits+indiv_scores', 'amaps_only', 'amaps+indiv_scores']:
            raise ValueError(f'Unsupported mode {val}')
        self._mode = val

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """ Forward pass

        :param x: Input tensor
        :returns: (classifier prediction, [confidence score], activation maps)
        """
        # Forward pass computing classification and updating self.features
        logits = self.classifier(x)  # N x C
        # Compute activation maps and confidence values
        amaps, confidence = self.detector(self.features)
        if self._mode == 'logits+amaps':
            # During training, loss function needs all activations maps for all classes
            # (selection based on ground truth labels)
            return logits, amaps
        # For each element of the batch, find most probable class
        class_idx = logits.argmax(dim=1, keepdim=True)  # Shape N x 1
        # Expand index to N x 1 x P
        target_shape = [confidence.size(0), 1, confidence.size(2)]
        class_idx = class_idx[:, :, None].expand(target_shape)
        if self.detector.nclasses == 1:
            confidence = confidence[:, 0]
            amaps = amaps[:, 0]
        else:
            # Confidence scores of most probable class
            confidence = confidence.gather(dim=1, index=class_idx).squeeze(dim=1)  # Shape N x P
            # Expand index to N x 1 x P x H x W
            target_shape = [amaps.size(0), 1] + list(amaps.size()[2:])
            class_idx = class_idx[:, :, :, None, None].expand(target_shape)
            # Activations maps of most probable class
            amaps = amaps.gather(dim=1, index=class_idx).squeeze(dim=1)  # Shape N x P x H x W
        if self._mode == 'production':
            # Production mode: average class confidence scores
            confidence = confidence.mean(dim=1)  # Shape N
            return logits, confidence
        elif self._mode == 'logits+indiv_scores':
            return logits, confidence
        elif self._mode == 'amaps+indiv_scores':
            # Compatibility with Particul model
            return amaps, confidence
        elif self._mode == 'amaps_only':
            # Compatibility with Particul model
            return amaps
        raise ValueError(f'Unknown mode {self._mode}')

    @property
    def calibrated(self) -> bool:
        """ Return detector calibration status """
        return self.detector.calibrated

    @property
    def enable_confidence(self) -> bool:
        """ Compatibility with Particul detector: enable/disable confidence measure """
        return self._mode in ['production', 'logits+indiv_scores', 'amaps+indiv_scores']

    @enable_confidence.setter
    def enable_confidence(self, val: bool) -> None:
        if val:
            self._mode = 'amaps+indiv_scores'
        else:
            self._mode = 'amaps_only'

    @property
    def enable_normalization(self) -> bool:
        """ Compatibility with Particul detector: enable/disable softmax normalization """
        return self.detector.enable_normalization

    @enable_normalization.setter
    def enable_normalization(self, val: bool) -> None:
        self.detector.enable_normalization = val

    @staticmethod
    def add_parser_options(parser: ArgumentParser) -> None:
        """ Add all options for the initialization of a ClassifierWithParticulOD2 model

        :param parser: Target argparse parser
        """
        parser.add_argument('--od2-classifier', type=str, required=True,
                            metavar='<path_to_model>',
                            help='Path to pretrained classifier.')
        parser.add_argument('--od2-detector', type=str, required=True,
                            metavar='<path_to_model>',
                            help='Path to ParticulOD2 detector.')

    @staticmethod
    def build_from_parser(args: Namespace) -> nn.Module:
        """ Build ClassifierWithParticulOD2 model from argparse arguments

        :param args: Command line argument
        """
        return ClassifierWithParticulOD2(
            classifier=torch_toolbox.load(args.od2_classifier, map_location=args.device),
            detector=ParticulOD2.load(args.od2_detector, map_location=args.device),
        )
