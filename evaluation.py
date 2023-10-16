import argparse
import json
import os
from scipy.stats import pearsonr,spearmanr,kendalltau

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--actual_path", type=str)
    parser.add_argument("--predicted_path", type=str)
    parser.add_argument("--target_metric", type=str)
    args = parser.parse_args()

    print("#"*10+"evaluation begins"+"#"*10)
    print("actual_path: ", args.actual_path)
    print("predicted_path: ", args.predicted_path)
    a={}
    with open(args.actual_path, 'r') as r:
        a_bank = json.loads(r.read())

    for qid in a_bank.keys():
        a[qid]=float(a_bank[qid][args.target_metric])

    p={}

    with open(args.predicted_path, 'r') as r:
        for line in r:
            try:
                qid, p_value = line.rstrip().split("\t")
            except:
                qid = line.rstrip().split("\t")[0]
                p[qid] = 0.0
                continue
            try:
                p[qid]=float(p_value)
            except:
                p[qid] = 0.0

    a_list = []
    p_list = []

    for qid in a.keys():
        a_list.append(a[qid])
        p_list.append(p[qid])

    print(f"#queries in actual performance: {len(a)}\n#queries in predicted performance: {len(p)}")

    pearson_coefficient, pearson_pvalue = pearsonr(a_list, p_list)
    kendall_coefficient, kendall_pvalue = kendalltau(a_list, p_list)

    result_dict = {"Pearson": round(pearson_coefficient, 3),
                   "Kendall": round(kendall_coefficient, 3),
                   "#q_actual": len(a),
                   "#q_predicted": len(p),
                   "P_pvalue": pearson_pvalue,
                   "K_pvalue": kendall_pvalue}

    print(result_dict)