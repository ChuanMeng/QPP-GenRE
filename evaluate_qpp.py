import argparse
import json
import os
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
from collections import defaultdict
import glob

def calculate_sARE(rank_1, rank_2, Q):
    return np.abs(rank_1-rank_2)/Q

def calculate_sMARE(list_1, list_2):
    assert len(list_1)==len(list_2)
    Q = len(list_1)
    sARE_scores = []

    for i in range(Q):
        sARE_scores.append(calculate_sARE(list_1[i], list_2[i], Q))

    return sum(sARE_scores)/Q

def evaluation(ap_path=None, pp_path=None, target_metric="ndcg@3"):
    print("#"*10+"evaluation begins"+"#"*10)
    print("ap_path: ", ap_path)
    print("pp_path: ", pp_path)

    ap={}
    with open(ap_path, 'r') as r:
        ap_bank = json.loads(r.read())

    for qid in ap_bank.keys():
        ap[qid]=float(ap_bank[qid][target_metric])

    pp={}
    with open(pp_path, 'r') as r:
        for line in r:
            try:
                qid, pp_value = line.rstrip().split("\t")
            except:
                qid = line.rstrip().split("\t")[0]
                if qid in ap:
                    pp[qid] = 0.0
                    continue

            try:
                if qid in ap:
                    pp[qid]=float(pp_value)
            except:
                if qid in ap:
                    pp[qid] = 0.0


    ap_list = []
    pp_list = []

    for qid in ap.keys():
        ap_list.append(ap[qid])
        pp_list.append(pp[qid])

    print(f'sanity check for {target_metric}: {round(np.mean(ap_list),3)}')
    print(f"len_ap: {len(ap)}, len_pp: {len(pp)}")
    print(f"ap's first 5 {ap_list[:5]}")
    print(f"pp's first 5 {pp_list[:5]}")

    pearson_coefficient, pearson_pvalue = pearsonr(ap_list, pp_list)
    kendall_coefficient, kendall_pvalue = kendalltau(ap_list, pp_list)
    spearman_coefficient, spearman_pvalue = spearmanr(ap_list, pp_list)

    # sMARE
    sorted_ap = list(sorted(ap.items(), key=lambda x: x[1]))
    sorted_pp = list(sorted(pp.items(), key=lambda x: x[1]))

    sorted_ap = list(map(lambda x:x[0], sorted_ap))
    sorted_pp = list(map(lambda x:x[0], sorted_pp))

    ap_qrank = []
    pp_qrank = []

    for qid in sorted_ap:
        ap_qrank.append(sorted_ap.index(qid))
        pp_qrank.append(sorted_pp.index(qid))

    smare = calculate_sMARE(pp_qrank, ap_qrank)

    result_dict = {"Pearson": round(pearson_coefficient, 3),
                   "Kendall": round(kendall_coefficient, 3),
                   "Spearman": round(spearman_coefficient, 3),
                   "sMARE": round(smare, 3),
                   "len_ap": len(ap),
                   "len_pp": len(pp),
                   "P_pvalue": pearson_pvalue,
                   "K_pvalue": kendall_pvalue,
                   "S_pvalue": spearman_pvalue}

    print(result_dict)

    return result_dict


def evaluation_epochs(ap_path=None, pp_path=None, target_metric="ndcg@3", epoch_num=10):
    for epoch in range(1, epoch_num + 1):
        print(f"Evaluation on epoch {epoch} in terms of {target_metric}.")
        result_dict = evaluation(ap_path=ap_path, pp_path=pp_path + "-" + str(epoch),target_metric=target_metric)
        print(result_dict)

        with open(pp_path + "." + target_metric, 'a+', encoding='utf-8') as w:
            w.write(str(epoch) + ": " + str(result_dict) + os.linesep)

def evaluation_glob(ap_path=None, pattern=None, target_metrics=["ndcg@3", "ndcg@100", "recall@100", "map@100"]):
    for target_metric in target_metrics:
        for pp_path in sorted(glob.glob(pattern)):
            name = pp_path.split("/")[-1]
            dataset = pp_path.split("/")[-1].split(".")[0]
            output_path ="/".join(pattern.split("/")[:-1])
            pattern_name = pattern.split("/")[-1]

            result_dict = evaluation(ap_path=ap_path, pp_path=pp_path, target_metric=target_metric)

            with open(f"{output_path}/result.{pattern_name}", 'a+', encoding='utf-8') as w:
                name_=f"{name}.{target_metric}:"
                w.write(f"{name_.ljust(85, ' ')} {str(result_dict)}{os.linesep}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--ap_path", type=str)
    parser.add_argument("--pp_path", type=str)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--epoch_num", type=int, default=None)
    parser.add_argument("--target_metric", type=str)
    parser.add_argument("--target_metrics", nargs='+')
    args = parser.parse_args()

    if args.epoch_num is not None:
        result = evaluation_epochs(args.ap_path, args.pp_path, target_metric=args.target_metric, epoch_num=args.epoch_num)
    elif args.pattern is not None:
        evaluation_glob(args.ap_path, args.pattern, target_metrics=args.target_metrics)
    else:
        result = evaluation(args.ap_path, args.pp_path, target_metric=args.target_metric)


