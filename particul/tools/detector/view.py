#!/usr/bin/env python3
import torch_toolbox
from particul.visualization.upsampling import upsampling
from particul.visualization.gradients import saliency, integrated_gradients, smooth_grads
from particul.visualization.utils import (
    apply_jet_heatmap,
    apply_mask,
    add_textbox,
    add_padding,
    extract_region,
)
from torch_toolbox.visualization.image_visualizer import ImageVisualizer
from torch_toolbox.datasets.dataset import dataset_add_parser_options, datasets_from_parser
from torch_toolbox.datasets.preprocessing import preprocessing_from_parser, split_transform
from math import ceil
import sys
import numpy as np
from argparse import RawTextHelpFormatter
import argparse

methods_dict = {
    'upsampling': upsampling,
    'saliency': saliency,
    'integrated_gradients': integrated_gradients,
    'smooth_grads': smooth_grads,
}

default_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                  [255, 0, 255], [0, 255, 255], [128, 0, 255], [255, 0, 128],
                  [128, 128, 255], [255, 255, 255]]

# Parse command line
ap = argparse.ArgumentParser(
    description='Visualize patterns on a dataset.',
    formatter_class=RawTextHelpFormatter)
ap.add_argument('-m', '--model', required=True, type=str,
                metavar='<path_to_file>',
                help='Path to model.')
ap.add_argument('--methods', type=str, required=False, nargs='+',
                metavar='<name>',
                choices=['upsampling', 'saliency', 'integrated_gradients', 'smooth_grads'],
                help='List of methods to be used.')
ap.add_argument('--device', type=str, required=True,
                metavar='<device_id>',
                help='Target device for execution.')
ap.add_argument('--nimages', type=int, required=False,
                metavar='<num>',
                default=5,
                help='Number of images per screen.')
ap.add_argument('--post', type=str, required=False, nargs='+',
                metavar='<name>',
                choices=(['jet', 'mask', 'extract', 'jet-black', 'aggregate']),
                default=['jet'],
                help='List of post-processing methods (jet, mask or extract)')
ap.add_argument('--patterns', type=int, required=False, nargs='+',
                metavar='<index>',
                help='List of pattern indices (default: None = all).')
ap.add_argument('--polarity', type=str, required=False,
                metavar='<value>',
                default='absolute',
                choices=(['positive', 'negative', 'absolute']),
                help='Gradient polarity.')
ap.add_argument('--gaussian-ksize', type=int, required=False,
                metavar='<value>',
                default=5,
                help='Size of Gaussian filter kernel.')
ap.add_argument('--noise-ratio', type=float, required=False,
                metavar='<value>',
                default=0.2,
                help='SmoothGrad noise ratio.')
ap.add_argument('--disable-normalization', required=False, action='store_true',
                help='Disable min-max gradient normalization')
ap.add_argument('--image-index', type=int, required=False,
                metavar='<index>',
                default=0,
                help='Index of first image (default: 0).')
ap.add_argument('-v', '--verbose', required=False, action='store_true',
                help='Verbose mode')
ap.add_argument('-q', '--quiet', required=False, action='store_true',
                help='Quiet mode, used only in continous integration')

# Add dataset options
dataset_add_parser_options(ap)
args = ap.parse_args()

# Temporarily disable image preprocessing in order to recover original images
preprocess, _ = preprocessing_from_parser(args)
args.preprocessing = []
transform, reshape, preprocess = split_transform(preprocess[0])

# Load model
model = torch_toolbox.load(args.model, map_location=args.device)

if model.__class__.__name__ in ['ParticulCNet', 'ParticulPNet']:
    model = model.extractor
    model.enable_features = False

# Set evaluation mode
model.eval()

# Pattern indices
ipatterns = args.patterns if args.patterns is not None else range(model.npatterns)
npatterns = len(ipatterns)

# Extend colors
colors = default_colors * (ceil(npatterns / len(default_colors)))

methods = [methods_dict[m] for m in args.methods] if args.methods else []

# Grid
nrows = args.nimages
ncols = npatterns * len(methods) * len(args.post) + 1
if 'aggregate' in args.post:
    ncols -= (npatterns - 1) * len(methods)

# Overwrite batch size and load dataset (ignore validation/test sets)
args.batch_size = 1
dataset, _, _ = datasets_from_parser(args)


def img_generator():
    res = []
    nimages = 0
    for iidx, (original, _) in enumerate(dataset):
        if iidx < args.image_index:
            continue
        if isinstance(original, list):
            original = original[0]
        # Keep track of the original image, after transform
        if transform is not None:
            original = transform(original)
        img = reshape(original) if reshape is not None else original
        res.append(img)
        nimages += 1
        if methods:
            # Compute saliency and confidence maps using all methods
            patterns, confidence = zip(*[
                method(
                    model=model,
                    img=img,
                    index=ipatterns,
                    preprocess=preprocess,
                    device=args.device,
                    verbose=args.verbose,
                    polarity=args.polarity,
                    gaussian_ksize=args.gaussian_ksize,
                    normalize=not args.disable_normalization,
                    noise=args.noise_ratio,
                ) for method in methods
            ])
            # Confidence (if any) should be the same for all methods
            confidence = confidence[0]

            # Text box location
            box_w, box_h = img.width, int(img.height * 0.15)
            box_fontsize = box_h - 2
            box_x, box_y = 0, img.height

            # Interleave methods and post processing
            nmethods = len(methods)
            for pidx in range(npatterns):
                processed = []
                for midx in range(nmethods):
                    pattern = patterns[midx][pidx]
                    if 'jet' in args.post:
                        processed.append(apply_jet_heatmap(img, pattern))
                    if 'jet-black' in args.post:
                        shape = list(pattern.shape) + [3]
                        processed.append(apply_jet_heatmap(np.zeros(shape), pattern))
                    if 'mask' in args.post:
                        processed.append(apply_mask(img, pattern, colors[pidx]))
                    if 'extract' in args.post:
                        # Extract pattern from original to preserve details
                        processed.append(extract_region(original, pattern))
                if confidence is not None:
                    score = confidence[pidx]
                    # Pad all images
                    processed = [add_padding(pattern, 0, box_h, 0, 0) for pattern in processed]
                    processed = [add_textbox(pattern, f'Conf: {int(round(score * 100))}%',
                                             box_x, box_y, box_w, box_h, box_fontsize)
                                 for pattern in processed]
                res += processed
            if 'aggregate' in args.post:
                for method_patterns in patterns:
                    # Aggregate all patterns
                    aggr = np.max(np.array(method_patterns), axis=0)
                    res.append(apply_mask(img, aggr, keep_largest=False))
        if nimages == nrows:
            yield res
            res = []
            nimages = 0


if args.quiet:
    # Dummy call to generator
    next(img_generator())
    sys.exit(0)

ImageVisualizer(
    img_generator(),
    nrows=nrows,
    ncols=ncols,
    wscreen=1500,
    hscreen=900,
    focus=None,
    use_col_textboxes=False,
    use_row_textboxes=False,
    verbose=False
)
