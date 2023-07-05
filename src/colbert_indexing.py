import argparse
import sys
sys.path.insert(0, "ColBERT/")

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, required=True, help="the dataset path")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="the name of the dataset")

    args = parser.parse_args()
    
    experiment = f"{args.dataset}_wiki"
    index_name = f"{args.dataset}.2bits"

    checkpoint = "colbert-ir/colbertv2.0"

    with Run().context(RunConfig(nranks=1, experiment=experiment)):
        config = ColBERTConfig(doc_maxlen=300, nbits=2, kmeans_niters=4)
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=args.path, overwrite=True)


