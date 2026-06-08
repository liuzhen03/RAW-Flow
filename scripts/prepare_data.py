#!/usr/bin/env python
import argparse
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

SCRIPT_MAP = {
    ("fivek", "train"): "data._processing.fivek_train",
    ("fivek", "eval"): "data._processing.fivek_eval",
    ("pascalraw", "train"): "data._processing.pascalraw_train",
    ("pascalraw", "eval"): "data._processing.pascalraw_eval",
}

def main():
    parser = argparse.ArgumentParser(description="Prepare RawFlow training/evaluation data")
    parser.add_argument("--dataset", required=True, choices=["fivek", "pascalraw"])
    parser.add_argument("--split", required=True, choices=["train", "eval"])
    parser.add_argument("--exiftool", default=None, help="Path to exiftool executable")
    parser.add_argument("--pascalraw-splits", default=None, help="Directory with PASCALRAW train.txt/test.txt")
    args, extra = parser.parse_known_args()

    if args.exiftool:
        os.environ["EXIFTOOL"] = args.exiftool
    if args.pascalraw_splits:
        os.environ["PASCALRAW_SPLITS"] = args.pascalraw_splits

    module = SCRIPT_MAP[(args.dataset, args.split)]
    sys.argv = [module] + extra
    runpy.run_module(module, run_name="__main__")

if __name__ == "__main__":
    main()
