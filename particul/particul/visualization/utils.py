import torch
import matplotlib.cm as cm
from skimage.filters import threshold_otsu
import numpy as np
from typing import List, Optional, Tuple, Union
from PIL import Image
from PIL import ImageDraw, ImageFont
import os


def migrate(
        T: torch.Tensor
) -> np.array:
    """ Copy data from a tensor into a numpy array

    :param T: Tensor
    """
    return T.detach().cpu().numpy().copy()


def polarity_and_collapse(
        array: np.array,
        polarity: Optional[str] = None,
        avg_chan: Optional[int] = None,
) -> np.array:
    """ Apply polarity filter (optional) followed by average over channels (optional)

    :param array: Target
    :param polarity: Polarity (positive, negative, absolute)
    :param avg_chan: Dimension across which channels are averaged
    """
    assert polarity in [None, 'positive', 'negative', 'absolute'], f'Invalid polarity {polarity}'

    # Polarity first
    if polarity == 'positive':
        array = np.maximum(0, array)
    elif polarity == 'negative':
        array = np.abs(np.minimum(0, array))
    elif polarity == 'absolute':
        array = np.abs(array)

    # Channel average
    if avg_chan is not None:
        array = np.average(array, axis=avg_chan)
    return array


def normalize_min_max(array: np.array) -> np.array:
    """ Perform min-max normalization of a numpy array

    :param array: Target
    """
    vmin = np.amin(array)
    vmax = np.amax(array)
    # Avoid division by zero
    return (array - vmin) / (vmax - vmin + np.finfo(np.float32).eps)


def tojet(array: np.array) -> np.array:
    """ Returns a JET colormap from a given np.array

    :param array: 2D array
    :return: Corresponding JET color map.

    Note: If array values are outside of [0,1], performs min-max Normalization
    """
    assert len(array.shape) == 2, 'tojet function requires 2D input array'

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    return (jet_colors[np.uint8(array * 255)] * 255).astype(np.uint8)


def get_largest_component(
        array: np.array
) -> np.array:
    """ Given a boolean array, return an array containing the largest region of
    positive values.

    :param array: Input array
    """
    # Early abort
    if (array == False).all():
        return array

    # Find largest connected component in a single pass
    ncols = array.shape[1]
    cmax = None
    smax = 0
    active = []
    for y in range(array.shape[0]):
        marked = array[y]
        processed = []
        for comp in active:
            # Recover last row activation
            cprev = comp[y - 1]
            # Check possible connections with current column
            cexp = np.convolve(cprev, np.array([True, True, True]))[1:-1]
            # Is there a match?
            match = np.logical_and(array[y], cexp)
            if (match != False).any():
                # Update marked (untick elements in match)
                marked = np.logical_and(marked, np.logical_not(match))
                comp[y] = match
                # Check merge condition
                merged = False
                for pidx in range(len(processed)):
                    if (np.logical_and(processed[pidx][y], match) != False).any():
                        merged = True
                        processed[pidx] = np.logical_or(processed[pidx], comp)
                        break
                if not merged:
                    processed.append(comp)
            else:
                # End of component
                size = np.sum(comp)
                if size > smax:
                    smax = size
                    cmax = comp
        active = processed

        # Init new components using unmarked elements
        i = 0
        while i < ncols:
            if marked[i]:
                # New component (all False)
                comp = np.zeros(array.shape) > 1.0
                # Extend component inside the current row
                while i < ncols and marked[i]:
                    comp[y, i] = True
                    i += 1
                active.append(comp)
            else:
                i += 1
    # Check last active components
    for comp in active:
        size = np.sum(comp)
        if size > smax:
            smax = size
            cmax = comp
    return cmax


def apply_jet_heatmap(
        img: Image.Image,
        grad: np.array,
        alpha: Optional[float] = 0.6,
) -> Image.Image:
    """ Apply jet heatmap on an image

    :param img: Source image
    :param grad: Gradient map
    :param alpha: Fusion parameter
    """
    jet_mask = Image.fromarray(tojet(grad))
    jet_mask = np.array(jet_mask)
    masked_img = (jet_mask * alpha + np.array(img) * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(masked_img)


def apply_mask(
        img: Image.Image,
        mask: np.array,
        value: Optional[List[float]] = None,
        alpha: Optional[float] = 0.5,
        keep_largest: Optional[bool] = True,
) -> Image.Image:
    """ Return image with colored mask cropped to Otsu's threshold

    :param img: Source image
    :param mask: Image mask
    :param value: Mask color
    :param alpha: Overlay intensity
    :param keep_largest: Keep largest connected component only
    """
    if value is None:
        value = [255.0, 0, 0]
    # Apply overlay intensity
    value = [v * alpha for v in value]
    # Apply threshold
    mask = mask > threshold_otsu(mask)
    if keep_largest:
        # Keep only one connected component
        mask = get_largest_component(mask)

    M = (np.array(img) + value) * np.expand_dims(mask, axis=-1)
    vmax = np.amax(M)
    if vmax > 255:
        M = np.round(M / vmax * 255)
    M += np.array(img) * np.expand_dims(1 - mask, axis=-1)
    return Image.fromarray(M.astype(np.uint8))


def extract_region_bbox(
        img: Image.Image,
        mask: Union[np.array, List[np.array]],
        keep_largest: Optional[bool] = True,
        min_shape: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = 1.0,
) -> Optional[Tuple[int, int, int, int]]:
    """ Return image cropped to the region of mask values higher than Otsu's threshold

    :param img: Source image
    :param mask: Image mask or list of masks
    :param keep_largest: Keep largest connected component only
    :param min_shape: Minimum shape (<width>,<height>) of extracted region (wrt mask dimension)
    :param scale_factor: Rescaling factor of extracted region
    :return: Bounding box of extracted region based on mask (if possible), None otherwise
    """
    if type(mask) != list:
        mask = [mask]

    # Apply Otsu's threshold
    mask = [m > threshold_otsu(m) for m in mask]
    if keep_largest:
        # Keep only one connected component
        mask = [get_largest_component(m) for m in mask]

    # Aggregate masks
    mask = np.max(np.array(mask), axis=0)

    # Early abort
    if (mask == False).all():
        return None

    # Find bounding box
    H = mask.shape[0]
    W = mask.shape[1]
    for x_min in range(W):
        if (mask[..., x_min] != False).any():
            break
    for x_max in reversed(range(W)):
        if (mask[..., x_max] != False).any():
            break
    for y_min in range(H):
        if (mask[y_min] != False).any():
            break
    for y_max in reversed(range(H)):
        if (mask[y_max] != False).any():
            break

    # Apply scale factor if any
    if scale_factor != 1.0:
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        nbbox_w = int(bbox_w * scale_factor)
        nbbox_h = int(bbox_h * scale_factor)
        x_min = max(0, x_min - int((nbbox_w - bbox_w) / 2))
        x_max = min(W, x_min + nbbox_w)
        y_min = max(0, y_min - int((nbbox_h - bbox_h) / 2))
        y_max = min(H, y_min + nbbox_h)

    # Adjust bounding box shape if necessary
    if min_shape:
        # Adjust min_shape to mask dimension if necessary
        bbox_w = min(min_shape[0], W)
        bbox_h = min(min_shape[1], H)
        if (x_max - x_min < bbox_w) or (y_max - y_min < bbox_h):
            # Get current box center
            x_c = (x_min + x_max) / 2
            y_c = (y_min + y_max) / 2
            # Adjust top-left corner
            x_min = max(0, x_c - bbox_w / 2)
            y_min = max(0, y_c - bbox_h / 2)
            x_min -= max(x_min + bbox_w + 1 - W, 0)
            y_min -= max(y_min + bbox_h + 1 - H, 0)
            x_max = x_min + bbox_w
            y_max = y_min + bbox_h

    # Rescale (image might have a different shape than mask)
    x_min = int(x_min * img.width / W)
    x_max = int(x_max * img.width / W)
    y_min = int(y_min * img.height / H)
    y_max = int(y_max * img.height / H)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1
    return x_min, y_min, x_max, y_max


def extract_region(
        img: Image.Image,
        mask: Union[np.array, List[np.array]],
        keep_largest: Optional[bool] = True,
        keep_shape: Optional[bool] = True,
        skip_empty_mask: Optional[bool] = False,
        min_shape: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = 1.0,
) -> Image.Image:
    """ Return image cropped to the region of mask values higher than Otsu's threshold

    :param img: Source image
    :param mask: Image mask or list of masks
    :param keep_largest: Keep largest connected component only
    :param keep_shape: Resize to mask shape after crop
    :param skip_empty_mask: Return None for empty masks
    :param min_shape: Minimum shape (<width>,<height>) of extracted region (wrt mask dimension)
    :param scale_factor: Rescaling factor of extracted region
    :return: Extracted region based on mask (if possible), original image or None otherwise
    """
    bbox = extract_region_bbox(img, mask, keep_largest, min_shape, scale_factor)
    if bbox is None:
        if skip_empty_mask:
            return None
        if keep_shape:
            return img.resize((mask.shape[1], mask.shape[0]))
        return img
    dst = img.crop(bbox)
    if keep_shape:
        dst = dst.resize((mask.shape[1], mask.shape[0]))
    return dst


def add_padding(
        img: Image.Image,
        top: int,
        bottom: int,
        right: int,
        left: int,
        color: Tuple[int] = (255,255,255),
) -> Image.Image:
    """
    Add margin around image.

    :param img: Source image
    :param top: Number of pixel to add
    :param bottom: Number of pixel to add
    :param right: Number of pixel to add
    :param left: Number of pixel to add
    :param color: Padding color
    :returns: Padded image
    """
    res = Image.new(img.mode, (img.width+right+left, img.height+top+bottom), color)
    res.paste(img, (left, top))
    return res

def add_textbox(
        img: Image.Image,
        content: str,
        x: Optional[int] = 180,
        y: Optional[int] = 200,
        w: Optional[int] = 44,
        h: Optional[int] = 24,
        font_size: Optional[int] = 20,
        padding: bool= False,
) -> Image.Image:
    """ Add textbox to image

    :param img: Source image
    :param content: Text content
    :param x: Box coordinate
    :param y: Box coordinate
    :param w: Box width
    :param h: Box height
    :param font_size: Font size
    :param padding: Add padding prior to insert text
    """
    draw = ImageDraw.Draw(img)
    draw.rectangle((x, y, x + w, y + h), fill='white')
    # Get path to font
    path = os.path.dirname(os.path.abspath(__file__)) + '/fonts/arial.ttf'
    font = ImageFont.truetype(path, font_size)
    draw.text((x + w / 2, y + h / 2), content, fill='black', anchor='mm', font=font)
    return img
