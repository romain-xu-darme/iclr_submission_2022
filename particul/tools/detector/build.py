#!/usr/bin/env python3
import torch
import torch_toolbox
from particul.detector.model import Particul
import argparse

# Parse command line
ap = argparse.ArgumentParser(
    description='Build a Particul model.')
ap.add_argument('-m', '--model', required=True, type=str,
                metavar='<path_to_file>',
                help='Path to model.')
ap.add_argument('-s', '--source', required=False, type=str,
                metavar='<path_to_file>',
                help='Path to source model (from a previous code version).')
ap.add_argument('-v', '--verbose', action='store_true',
                help="Verbose mode")
Particul.add_parser_options(ap)
args = ap.parse_args()

model = Particul.build_from_parser(args)

if args.source is not None:
    # Load state dict
    ref_state = torch_toolbox.load(args.source).state_dict()
    if 'detectors.0.calibration' in ref_state.keys():
        # Reset calibration values (changed from normal to logistic distribution)
        # Split calibration values
        for ipattern in range(args.particul_npatterns):
            ref_state[f'detectors.{ipattern}.confidence._mean'] = torch.FloatTensor([-1.0 * float('inf')])
            ref_state[f'detectors.{ipattern}.confidence._mean'] = torch.FloatTensor([1.0])
        model.load_state_dict(ref_state, strict=False)

torch_toolbox.save(model, args.model)
if args.verbose: print(model)
