import argparse
import json
import os
import numpy as np
from collections import defaultdict
import glob
import pytrec_eval
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report


def evaluation(args):

    target_names = args.multi_text if not args.binary else args.binary_text
    labels = args.multi_num if not args.binary else args.binary_num
    place = 3

    with open(args.qrels_true_dir, 'r') as r_qrels:
        qrels_true = pytrec_eval.parse_qrel(r_qrels)

    with open(args.qrels_pred_dir, 'r') as r_qrels:
        qrels_pred = pytrec_eval.parse_qrel(r_qrels)

    true_list=[]
    pred_list=[]

    for qid in qrels_true.keys():
        for pid in qrels_true[qid].keys():
            true_list.append(qrels_true[qid][pid] if not args.binary else (1 if qrels_true[qid][pid] in args.multi_num_pos else 0))
            pred_list.append(qrels_pred[qid][pid] if args.pre_is_binary or not args.binary else (1 if qrels_pred[qid][pid] in args.multi_num_pos else 0))

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels_true_dir", type=str)
    parser.add_argument("--qrels_pred_dir", type=str)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--binary", action='store_true')
    parser.add_argument("--pre_is_binary", action='store_true')
    args = parser.parse_args()

    args.dataset_class = args.qrels_true_dir.split("/")[-3]
    #assert args.dataset_class == args.qrels_pred_dir.split("/")[-3]

    if "msmarco" in args.dataset_class:
        args.multi_text = ["Perfectly relevant", "Highly relevant", "Related", "Irrelevant"]
        args.binary_text = ["Relevant", "Not relevant"]
        args.multi_num =  [3, 2, 1, 0]
        args.multi_num_pos = [2, 3]
        args.binary_num = [1, 0]

    elif "ikat" == args.dataset_class:
        args.multi_text = ["Fully meets", "Highly meets", "Moderately meets", "Slightly meets", "Fails to meet"]
        args.binary_text = ["Relevant", "Not relevant"]
        args.multi_num =  [4, 3, 2, 1, 0]
        args.multi_num_pos = [2,3,4]
        args.binary_num = [1, 0]


    result = evaluation(args)


