import argparse
import os
import sys
import pytrec_eval
import json
from collections import defaultdict
import numpy as np
import math

def evaluation(args, n):
    mapping = {"ndcg_cut_10": "ndcg@10",
               "mrr_10": "mrr@10",
               "map_cut_100": "map@100",
               "recall_100": "recall@100",
               "recall_1000": 'recall@1000',
               "P_10": "precision@10",
               }

    with open(args.run_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    qrels_list = defaultdict(list)
    with open(args.qrels_path, "r") as r:
        qrels = r.readlines()
    for line in qrels:
        qid, _, pid, rel = line.split()
        qrels_list[qid].append({pid: int(rel)})

    qrels_dict = defaultdict(dict)
    for qid in qrels_list.keys():
        for pid_rel in qrels_list[qid][:n]:
            qrels_dict[qid].update(pid_rel)

    print(f"#queries in run {len(run)}\n#queries in qrels {len(qrels_dict)}")


    run_10 = {}
    for qid, did_score in run.items():
        sorted_did_score = [(did, score) for did, score in sorted(did_score.items(), key=lambda item: item[1], reverse=True)]
        run_10[qid] = dict(sorted_did_score[0:10])

    results = {}
    evaluator_general = pytrec_eval.RelevanceEvaluator(qrels_dict, {'ndcg_cut_10', 'map_cut_100', 'recall_100', 'recall_1000', 'P_10'})
    results_general = evaluator_general.evaluate(run)

    for qid, _ in results_general.items():
        results[qid]={}
        for measure, score in results_general[qid].items():
            results[qid][mapping[measure]] = score

    evaluator_rr = pytrec_eval.RelevanceEvaluator(qrels_dict, {'recip_rank'})
    results_rr_10 = evaluator_rr.evaluate(run_10)

    for qid, _ in results.items():
        results[qid][mapping["mrr_10"]] = results_rr_10[qid]['recip_rank']

    for measure in mapping.values():
        overall = pytrec_eval.compute_aggregated_measure(measure, [result[measure] for result in results.values()])
        print('{}: {:.4f}'.format(measure, overall))

    for measure in mapping.values():
        with open(f"{args.output_path}/{args.qrels_name}-n{n}-{measure}", 'w') as qpp_w:
            for qid in results.keys():
                qpp_w.write(f"{qid}\t{results[qid][measure]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_path', type=str, required=True)
    parser.add_argument('--qrels_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument("--n", nargs='+')
    args = parser.parse_args()
    args.qrels_name = args.qrels_path.split("/")[-1]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for n in args.n:
        print(f"***********juding depth{n}************")
        evaluation(args,int(n))
