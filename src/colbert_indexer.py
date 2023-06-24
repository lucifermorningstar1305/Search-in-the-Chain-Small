import sys
import argparse
import os

from ..search_in_the_chain.ColBERT.colbert.infra import Run, RunConfig, ColBERTConfig
from ..search_in_the_chain.ColBERT.colbert import Indexer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="checkpoint path of the colbert model")
    parser.add_argument("--name", "-a", type=str, required=True, help="sets the name for the colbert index")
    parser.add_argument("--data", "-d", type=str, required=True, help="the path of the data")
    parser.add_argument("--experiment", "-e", type=str, required=False, default="wiki", help="to the set the name of experiment for colbert")
    parser.add_argument("--nranks", "-n", type=int, required=False, default=1, help="sets the rank for the colbert")

    args = parser.parse_args()

    with Run().context(RunConfig(nranks=args.nranks, experiment=args.experiment)):

        config = ColBERTConfig(
            nbits = 2, 
            root="colbert-experiments"
        )

        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(name=args.name, collection=args.data)



        