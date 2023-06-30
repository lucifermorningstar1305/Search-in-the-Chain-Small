from flask import Flask, render_template, request
from functools import lru_cache
import math
import os
import sys
from dotenv import load_dotenv
import argparse

sys.path.append("..")

from ..search_in_the_chain.ColBERT.colbert.infra import Run, RunConfig, ColBERTConfig
from ..search_in_the_chain.ColBERT.colbert import Searcher


load_dotenv()
INDEX_NAME = os.getenv("INDEX_NAME")
INDEX_ROOT = os.getenv("INDEX_ROOT")
app = Flask(__name__)

#searcher = Searcher(index=f"{INDEX_ROOT}/{INDEX_NAME}")


@lru_cache(maxsize=1000000)
def api_search_query(query, k):
    print(f"Query={query}")
    if k == None: k = 10
    k = min(int(k), 100)
    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    passages = [searcher.collection[pid] for pid in pids]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]            
        d = {'text': text, 'pid': pid, 'rank': rank, 'score': score, 'prob': prob}
        topk.append(d)
    topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
    return {"query" : query, "topk": topk}

@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        print("API request count:", counter["api"])
        return api_search_query(request.args.get("query"), request.args.get("k"))
    else:
        return ('', 405)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", "-i", required=True, type=str, help="name of the colbert index of your dataset")
    args = parser.parse_args()

    # searcher = Searcher(index=f"/ColBERT-main/experiments/hotpotqa_wiki/indexes/hotpotqa_wiki.nbits=2")
    searcher = Searcher(index=args.index_name)

    counter = {"api" : 0}
    app.run("0.0.0.0", int(os.getenv("PORT")))

