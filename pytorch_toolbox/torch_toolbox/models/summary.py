#!/usr/bin/env python3
import torch_toolbox
from argparse import RawTextHelpFormatter
import argparse


def _main():
    # Parse command line
    ap = argparse.ArgumentParser(
        description='Print Torch model',
        formatter_class=RawTextHelpFormatter)
    ap.add_argument('-m', '--model', required=True, type=str,
                    metavar='<path_to_file>',
                    help='Path to model.')

    args = ap.parse_args()
    model = torch_toolbox.load(args.model, map_location='cpu')
    print(model)


if __name__ == '__main__':
    _main()
