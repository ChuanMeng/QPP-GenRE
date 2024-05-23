import argparse
import json
import os
import numpy as np
from collections import defaultdict
import glob
import pytrec_eval
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report


def evaluation(qrels_true_dir=None, qrels_pred_dir=None, binary=False, pre_is_binary = False):

    target_names = ["Perfectly relevant", "Highly relevant", "Related", "Irrelevant"] if not binary else ["Relevant", "Not relevant"]
    labels = [3, 2, 1, 0] if not binary else [1, 0]
    place = 3

    with open(qrels_true_dir, 'r') as r_qrels:
        qrels_true = pytrec_eval.parse_qrel(r_qrels)

    with open(qrels_pred_dir, 'r') as r_qrels:
        qrels_pred = pytrec_eval.parse_qrel(r_qrels)

    true_list=[]
    pred_list=[]

    for qid in qrels_true.keys():
        for pid in qrels_true[qid].keys():
            true_list.append(qrels_true[qid][pid] if not binary else (1 if qrels_true[qid][pid] in [2,3] else 0))
            pred_list.append(qrels_pred[qid][pid] if pre_is_binary or not binary else (1 if qrels_pred[qid][pid] in [2,3] else 0))

    cohen_kappa = cohen_kappa_score(true_list,pred_list)
    accuracy = accuracy_score(true_list,pred_list)
    report = classification_report(true_list, pred_list,
                                   labels=labels,
                                   output_dict=True,
                                   digits=place,
                                   target_names=target_names)

    matrix = confusion_matrix(true_list, pred_list, labels=labels)

    # sanity check
    assert accuracy == report['accuracy']
    for idx, class_name in enumerate(target_names):
        assert report[class_name]["support"]==matrix[idx].sum()

    # print something that is not easy to store
    print("confusion_matrix:\n",matrix.T)
    print(classification_report(true_list, pred_list,
                                labels=labels,
                                digits=place,
                                target_names=target_names))

    result_dict = {"cohen_kappa": round(cohen_kappa, place),
                   "accuracy": round(report['accuracy'], place),
                   "f1-score": round(report['macro avg']['f1-score'], place),
                   "precision": round(report['macro avg']['precision'], place),
                   "recall": round(report['macro avg']['recall'], place),
                   "num": report['macro avg']["support"]}

    print(result_dict)

    return result_dict



def evaluation_glob(ap_path=None, pattern=None):
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
    parser.add_argument("--qrels_true_dir", type=str)
    parser.add_argument("--qrels_pred_dir", type=str)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--binary", action='store_true')
    parser.add_argument("--pre_is_binary", action='store_true')
    args = parser.parse_args()

    if args.pattern is not None:
        evaluation_glob(args.qrels_true_dir, args.qrels_pred_dir, rgs.pattern)
    else:
        result = evaluation(args.qrels_true_dir, args.qrels_pred_dir, args.binary, args.pre_is_binary)


