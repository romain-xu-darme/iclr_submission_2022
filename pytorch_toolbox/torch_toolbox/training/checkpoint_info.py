#!/usr/bin/env python3
import torch
import argparse


def _main():
    # Parse command line
    ap = argparse.ArgumentParser(
        description='Display checkpoint informations.')
    ap.add_argument('-c', '--checkpoint', required=True, type=str,
                    metavar='<path_to_file>',
                    help='Path to checkpoint file.')
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    print(f"Epoch: {checkpoint['epoch']}\n"
          f"Metrics: {checkpoint['metrics']}")


if __name__ == '__main__':
    _main()
