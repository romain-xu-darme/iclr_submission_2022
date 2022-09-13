import torch.nn as nn
from .utils import migrate
from typing import List, Callable, Optional, Tuple
import PIL
from PIL import Image
from progress.bar import Bar
import numpy as np


def upsampling(
        model: nn.Module,
        img: Image.Image,
        index: List[int],
        preprocess: Callable,
        device: Optional[str] = 'cpu',
        verbose: Optional[bool] = False,
        **kwargs,
) -> Tuple[List[Image.Image], np.array]:
    """ Perform visualisation using bilinear upsampling

    :param model: Model
    :param img: Input image
    :param index: List of pattern indices
    :param preprocess: Preprocessing function
    :param device: Target device
    :param verbose: Show progress bar
    :return: List of pattern visualisations, confidence values
    """
    # Convert list of images to tensor
    tensor = preprocess(img)[None]
    tensor = tensor.to(device, non_blocking=True)

    # Compute activation maps
    if model.calibrated:
        model.enable_confidence = True
        amaps, confidence = model(tensor)
        confidence = migrate(confidence)[..., index]
    else:
        amaps, confidence = model(tensor), None
    amaps = migrate(amaps)

    # Permute channels
    amaps = np.moveaxis(amaps, 1, -1)[0]
    # amaps has shape H x W x P
    res = []
    bar = Bar('Computing upsampling', max=len(index), suffix='%(percent).1f%%') if verbose else None
    for pidx in index:
        pattern = Image.fromarray(amaps[..., pidx])
        # Resize pattern
        pattern = pattern.resize((img.width, img.height), resample=PIL.Image.BILINEAR)
        pattern = np.array(pattern)
        res.append(pattern)
        if verbose:
            bar.next()
    if verbose:
        bar.finish()
    return res, confidence
