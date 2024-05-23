import pandas as pd
import csv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str)
    args = parser.parse_args()

    with open(args.raw_data_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        df = pd.DataFrame(reader, columns=['turn_id', 'rewritten_utterance', 'response',  'pid', 'psg', 'rel', 'ptkb', 'set'])

    query={}
    ptkb={}
    corpus={}

    count_all = 0
    count_train = 0
    count_dev = 0
    count_test = 0
    count_none = 0

    rel_all ={"0":0,"1":0,"2":0,"3":0,"4":0}
    rel_train ={"0":0,"1":0,"2":0,"3":0,"4":0}
    rel_dev ={"0":0,"1":0,"2":0,"3":0,"4":0}
    rel_test = {"0":0,"1":0,"2":0,"3":0,"4":0}
    rel_none = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}

    with open("./datasets/ikat/qrels/ikat.qrels", "w") as w_all, open("./datasets/ikat/qrels/ikat-train.qrels", "w") as w_train, open("./datasets/ikat/qrels/ikat-dev.qrels", "w") as w_dev, open("./datasets/ikat/qrels/ikat-test.qrels", "w") as w_test:
        for idx in range(len(df)):
            qid = df.at[idx, "turn_id"]
            pid = df.at[idx, "pid"]
            rel = df.at[idx, "rel"]

            query[qid]=df.at[idx, "rewritten_utterance"]
            ptkb[qid] = df.at[idx, "ptkb"]
            corpus[pid] = df.at[idx, "psg"]

            rel_all[df.at[idx, "rel"]] += 1
            count_all+=1
            w_all.write(f"{qid} 0 {pid} {rel}\n")

            if df.at[idx, "set"] == "train":
                rel_train[df.at[idx, "rel"]]+=1
                count_train += 1
                w_train.write(f"{qid} 0 {pid} {rel}\n")
            elif df.at[idx, "set"] == "validation":
                rel_dev[df.at[idx, "rel"]] += 1
                count_dev += 1
                w_dev.write(f"{qid} 0 {pid} {rel}\n")
            elif df.at[idx, "set"] == "test":
                rel_test[df.at[idx, "rel"]] += 1
                count_test += 1
                w_test.write(f"{qid} 0 {pid} {rel}\n")
            else:
                rel_none[df.at[idx, "rel"]] += 1
                count_none += 1
                assert df.at[idx, "set"] == "NONE", print(df.at[idx, "set"])

        print(count_all,count_train, count_dev, count_test, count_none)
        assert sum([count_train, count_dev, count_test, count_none])==len(df)==count_all

        print(rel_all)
        print(rel_train)
        print(rel_dev)
        print(rel_test)
        print(rel_none)

    print(set(list(df['set'])))
    print(f"total: {len(df)}")

    with open("./datasets/ikat/queries/ikat.queries-manual", "w") as w_q:
        for qid in query.keys():
            w_q.write(f"{qid}\t{query[qid]}\n")

    with open("./datasets/ikat/queries/ikat.queries-manual-ptkb", "w") as w_q:
        for qid in query.keys():
            if ptkb[qid]=="NONE":
                concat=f"{query[qid]}"
            else:
                concat = f"{ptkb[qid]} {query[qid]}"
            w_q.write(f"{qid}\t{concat}\n")

    with open("./datasets/ikat/queries/ikat.ptkb", "w") as w_ptkb:
        for qid in ptkb.keys():
            w_ptkb.write(f"{qid}\t{ptkb[qid]}\n")

    with open("./datasets/ikat/corpus/ikat.corpus", "w") as w_corpus:
        for pid in corpus.keys():
            w_corpus.write(f"{pid}\t{corpus[pid]}\n")

    print(len(query),len(ptkb),len(corpus))
