#!/usr/bin/env python3
from particul.od2.fnrd.fnrd import FNRDModel
import argparse
from argparse import RawTextHelpFormatter


def _main():
    ap = argparse.ArgumentParser(
        description='Extract the Major Function Region (MFR) of a trained network on a reference dataset.',
        formatter_class=RawTextHelpFormatter)
    FNRDModel.add_parser_options(ap, add_mfr=True)
    args = ap.parse_args()
    FNRDModel.compute_mfr_from_parser(args)


if __name__ == '__main__':
    _main()
