import sys
import argparse
import os

sys.path.append("..")

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", "-a", type=str, required=True, help="sets the name for the colbert index")
    parser.add_argument("--data", "-d", type=str, required=True, help="the path of the data")
    parser.add_argument("--experiment", "-e", type=str, required=False, default="wiki", help="to the set the name of experiment for colbert")

    args = parser.parse_args()

    with Run().context(RunConfig(nranks=1, experiment=args.experiment)):

        config = ColBERTConfig(
            nbits = 2, 
            root="./ColBERT/experiments/"
        )

        indexer = Indexer(checkpoint="./colbertv2.0", config=config)
        indexer.index(name=args.name, collection=args.data)



        