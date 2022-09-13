#!/usr/bin/env python3
import torch_toolbox
from particul.od2.model import ParticulOD2
import argparse


def _main():
    # Parse command line
    ap = argparse.ArgumentParser(
        description='Build a Particul-based Out-of-Distribution detector.')
    ap.add_argument('-m', '--model', required=True, type=str,
                    metavar='<path_to_file>',
                    help='Path to model.')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help="Verbose mode")
    ParticulOD2.add_parser_options(ap)
    args = ap.parse_args()

    model = ParticulOD2.build_from_parser(args)
    torch_toolbox.save(model, args.model)
    if args.verbose:
        print(model)


if __name__ == '__main__':
    _main()
