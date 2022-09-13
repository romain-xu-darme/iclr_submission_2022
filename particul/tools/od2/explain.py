#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from particul.od2.model import ClassifierWithParticulOD2
from torch_toolbox.visualization.image_visualizer import ImageVisualizer
from torch_toolbox.datasets.dataset import (
    dataset_add_parser_options,
    get_classification_dataset,
)
from torch_toolbox.datasets.preprocessing import (
    preprocessing_from_parser,
    split_transform,
)
from particul.visualization.upsampling import upsampling
from particul.visualization.gradients import saliency, integrated_gradients, smooth_grads
from particul.visualization.utils import extract_region, add_textbox, apply_mask
from typing import Callable, List
from PIL import Image
import random
import argparse
from argparse import RawTextHelpFormatter

methods_dict = {
    'upsampling': upsampling,
    'saliency': saliency,
    'integrated_gradients': integrated_gradients,
    'smooth_grads': smooth_grads,
}


def explain_image(
        img: Image.Image,
        true_label: int,
        model: nn.Module,
        ipatterns: List[int],
        transform: Callable,
        method: str,
        device: str,
        extract: bool,
        verbose: bool,
) -> List[Image.Image]:
    """ Explain single image.

    :param img: Target image
    :param true_label: True label of target image (if known)
    :param model: Model
    :param ipatterns: List of pattern index
    :param transform: Preprocessing function
    :param method: Name of extraction method
    :param device: Device
    :param extract: Extract image regions
    :param verbose: Verbose mode
    :return: List of images
    """
    # Split operations
    transform, reshape, preprocess = split_transform(transform)
    if transform is not None:
        img = transform(img)  # Potential random ops are done once
    reshaped_img = reshape(img) if reshape is not None else img

    # Perform inference
    test_tns = preprocess(reshaped_img).unsqueeze(dim=0).to(device)
    logits = model.classifier(test_tns)
    smax = torch.max(torch.softmax(logits, dim=1)).item()
    pred = torch.argmax(logits[0]).item()

    # Set model to "Particul" mode
    model.mode = "amaps+indiv_scores"

    # Recover test parts
    patterns, confidence = methods_dict[method](
        model=model,
        img=reshaped_img,
        index=ipatterns,
        preprocess=preprocess,
        device=device,
        verbose=verbose,
        polarity='absolute',
        gaussian_ksize=3,
        normalize=True,
        noise=0.2,
    )

    if extract:
        test_parts = [extract_region(
            img=img,  # Use original image size
            mask=r,
            keep_largest=True,
            keep_shape=True,
            skip_empty_mask=False,
            min_shape=(96, 96),
            scale_factor=1.0,
        ) for r in patterns]
    else:
        test_parts = [apply_mask(
            img=reshaped_img,
            mask=r,
        ) for r in patterns]

    # Compute textbox locations
    box_w, box_h = reshaped_img.width, int(reshaped_img.height * 0.1)
    box_fontsize = box_h - 2
    box_x, box_y = 0, reshaped_img.height - box_h

    final_parts = []
    for test_part, score in zip(test_parts, confidence):
        text = f'Conf: {int(round(score * 100))}%'
        part_img = add_textbox(test_part, text,
                               box_x, box_y, box_w, box_h, box_fontsize)
        final_parts.append(part_img)

    # Infos are printed on white image
    infos_img = Image.fromarray(np.ones((reshaped_img.height, reshaped_img.width, 3), dtype=np.uint8) * 255)
    message = f'Prediction: {pred} ({int(smax * 100)}%)\n'
    if true_label is not None:
        message += f'Ground truth: {true_label}\n'
    infos_img = add_textbox(infos_img, message,
                            0, img.height // 2, box_w, box_h, box_fontsize)

    return [img, infos_img] + final_parts


def _main():
    ap = argparse.ArgumentParser(
        description='Perform inference with explanation.',
        formatter_class=RawTextHelpFormatter,
    )
    ClassifierWithParticulOD2.add_parser_options(ap)
    ap.add_argument('-i', '--image', required=False, type=str, nargs='+',
                    metavar='<path_to_file>',
                    help='Path to test images.')
    group = ap.add_argument_group('Reference dataset')
    dataset_add_parser_options(group)
    group.add_argument('--class-index', required=False, type=int, metavar='<index>',
                       help='Select only images from a given class')
    ap.add_argument('--nimages', required=False, type=int,
                    default=5,
                    metavar='<num>',
                    help='Max number of images per window.')
    ap.add_argument('--method', required=False, type=str,
                    choices=methods_dict.keys(),
                    default='smooth_grads',
                    metavar='<algorithm>',
                    help='Part extraction method applied on test image.')
    ap.add_argument('--patterns', type=int, required=False, nargs='+',
                    metavar='<index>',
                    help='List of pattern indices (default: None = all).')
    ap.add_argument("--extract-regions", action='store_true',
                    help="Extract image region instead of apply pattern masks.")
    ap.add_argument('--device', type=str, required=False,
                    metavar='<device_id>',
                    default='cpu',
                    help='Target device for execution.')
    ap.add_argument('-v', "--verbose", action='store_true',
                    help="Verbose mode")
    args = ap.parse_args()

    # Extract image preprocessing in order to recover original images
    transform, target_transform = preprocessing_from_parser(args)
    transform = transform[-1]  # Testset preprocessing

    # Load model, set evaluation mode and output mode
    model = ClassifierWithParticulOD2.build_from_parser(args)
    model.eval()

    for p in range(model.detector.npatterns):
        model.detector.particuls[0].detectors[p].confidence.debug = True

    if not model.calibrated:
        raise ValueError('Model not calibrated yet')

    ipatterns = args.patterns if args.patterns is not None \
        else range(model.detector.npatterns)

    def img_generator():
        if args.image:
            testset = [(Image.open(path), None) for path in args.image]
            random_walk = False
        else:
            testset = get_classification_dataset(
                name=args.dataset_name,
                root=args.dataset_location,
                split='test',
                transform=None,
                target_transform=None,
                download=True,
                verbose=False
            )
            args.seed = args.seed or 0
            random.seed(args.seed)
            random_walk = True

        accumulator = []
        for i in range(len(testset)):
            iidx = random.randint(0, len(testset) - 1) if random_walk else i
            test_img, label = testset[iidx]
            if args.class_index is not None and label != args.class_index:
                continue
            explanation = explain_image(
                img=test_img,
                true_label=label,
                model=model,
                ipatterns=ipatterns,
                transform=transform,
                method=args.method,
                device=args.device,
                extract=args.extract_regions,
                verbose=args.verbose,
            )
            accumulator += explanation
            if len(accumulator) == args.nimages * (2 + len(ipatterns)):
                yield accumulator
                accumulator = []
        if accumulator:
            yield accumulator

    # Show prediction with pattern detection
    ImageVisualizer(
        img_generator(),
        nrows=args.nimages,
        ncols=len(ipatterns) + 2,
        wscreen=1500,
        hscreen=900,
        focus=None,
        use_col_textboxes=False,
        use_row_textboxes=False,
        verbose=False
    )


if __name__ == '__main__':
    _main()
