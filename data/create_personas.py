"""
"""
import os
import argparse
import jsonlines

from datasets import load_dataset

def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, default="files")
    args = parser.parse_args()
    main(args)

