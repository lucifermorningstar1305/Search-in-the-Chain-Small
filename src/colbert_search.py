from typing import Any, List, Tuple, Union, Dict

import math
import argparse
import sys
import os

sys.path.insert(0, "ColBERT/")


from flask import Flask, request
from functools import lru_cache
from dotenv import load_dotenv
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

app = Flask(__name__)

@lru_cache(maxsize=1_000_000)
def api_search_query(query:str, k:int) -> Dict[str, List]:
    print(f"Query : {query}")

    if k == None: k = 10
    k = min(int(k), 100)

    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]

    print(pids)
    passages = [searcher.collection[pid] for pid in pids]
    probs = [math.exp(score) for score in scores]
    probs = [prob/sum(probs) for prob in probs]

    top_k = list()

    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        txt = searcher.collection[pid]
        d = {"text": txt, 
             "pid" : pid, 
             "rank": rank, 
             "score": score, 
             "prob": prob}
        
        top_k.append(d)
    
    top_k = list(sorted(top_k, key=lambda x: (-1 * x['score'], x['pid'])))
    return {"query": query, 
            "topk":top_k}


@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        return api_search_query(request.args.get("query"), request.args.get("k"))
    else:
        return ('', 405)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, type=str, help="the dataset path")
    parser.add_argument("--dataset", "-d", required=True, type=str, help="the name of the dataset")

    args = parser.parse_args()

    experiment = f"{args.dataset}_wiki"
    index_name = f"{args.dataset}.2bits"

    checkpoint = "colbert-ir/colbertv2.0"
    searcher = None
    with Run().context(RunConfig(experiment=experiment)):
        searcher = Searcher(index=index_name, collection=args.path)

    app.run("0.0.0.0", 50003)
    
    