import os
import sys
import argparse

from rich.progress import track

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True, type=str, help="the tsv to be cleaned")

    args = parser.parse_args()
    
    with open(args.file, "r") as fp:
        data = fp.readlines()

    for idx in track(range(len(data))):
        data[idx] = data[idx].replace("\ufeff0", "0")
    
    with open(args.file, "w") as fp:
        fp.writelines(data)
