# Query Performance Prediction using Relevance Judgments Generated by Large Language Models
This repository supports the paper "Query Performance Prediction using Relevance Judgments Generated by Large Language Models".
In this paper, we propose `QPP-GenRE`, which first automatically generates relevance judgments for a ranked list for a given query, and then regard the generated relevance judgments as pseudo labels to compute different IR evaluation measures; we fine-tune LLaMA-7B to automatically generate relevance judgments. 
  
This repository contains the following 5 components:
1. Installation
2. Inference using fine-tuned LLaMA
3. Fine-tuning LLaMA
4. In-context learning using LLaMA
5. Evaluation

## 1. Installation

### Install dependencies
```bash
pip install -r requirements.txt
```
### Download datasets
First download `dataset.zip` (containing queries, run files, qrels files and so on) from [here](https://drive.google.com/file/d/1d_bEofABPmnQKdHk-fdYT02tyzB4VBmI/view?usp=share_link), and then unzip it in the current directory.

Then, download MS MARCO V1 and V2 passage ranking collections from [Pyserini](https://github.com/castorini/pyserini):
```bash
wget -P ./datasets/msmarco-v1-passage/ https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.msmarco-v1-passage-full.20221004.252b5e.tar.gz --no-check-certificate
tar -zxvf  ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e.tar.gz -C ./datasets/msmarco-v1-passage/

wget -P ./datasets/msmarco-v2-passage/ https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a.tar.gz --no-check-certificate
tar -zxvf  ./datasets/msmarco-v2-passage/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a.tar.gz -C ./datasets/msmarco-v2-passage/
```

### Download the weights of LLaMA-7B
Please refer to the LLaMA [repository](https://github.com/facebookresearch/llama/tree/llama_v1) to fetch the weights of LLaMA-7B.
And follow the instructions from [here](https://huggingface.co/docs/transformers/main/model_doc/llama) to convert the original weights for the LLaMA-7B model to the Hugging Face Transformers format. 
Next, set your local path to the weights of LLaMA-7B (Hugging Face Transformers format) as a variable.
```bash
export LLAMA_7B_PATH={your path to the weights of LLaMA-7B (Hugging Face Transformers format)}
```

### Download the checkpoints of finetuned LLaMA-7B
We release ***the checkpoints of our finetuned LLaMA-7B*** for the reproducibility of the results reported in the paper.
Please download `checkpoint.zip` from [here](https://drive.google.com/file/d/1dGeJS0lJxMtZwGKrZaTRefe4TImxEQ1n/view?usp=share_link), and then unzip it in the current directory.

## 2. Inference using fine-tuned LLaMA

### Predicting the performance of BM25 on TREC-DL 19 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-2790 \
--query_path ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt  \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000 \
--output_path ./output/dl-19-passage \
```

### Predicting the performance of BM25 on TREC-DL 20 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-2790 \
--query_path ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt  \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000 \
--output_path ./output/dl-20-passage \
```

### Predicting the performance of BM25 on TREC-DL 21 
```bash
python judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-1860 \
--query_path ./datasets/msmarco-v2-passage/queries/dl-21-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v2-passage/runs/dl-21-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v2-passage/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a \
--qrels_path ./datasets/msmarco-v2-passage/qrels/dl-21-passage.qrels.txt \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v2-passage/runs/dl-21-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000 \
--output_path ./output/dl-21-passage \
```

### Predicting the performance of BM25 on TREC-DL 22 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-1860 \
--query_path ./datasets/msmarco-v2-passage/queries/dl-22-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v2-passage/runs/dl-22-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v2-passage/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a \
--qrels_path ./datasets/msmarco-v2-passage/qrels/dl-22-passage.qrels-withDupes.txt \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v2-passage/runs/dl-22-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-22-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000 \
--output_path ./output/dl-22-passage \
```

### Predicting the performance of ANCE on TREC-DL 19 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-2790 \
--query_path ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt  \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--qrels_path  ./output/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000 \
--output_path ./output/dl-19-passage \
```
### Predicting the performance of ANCE on TREC-DL 20 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-2790 \
--query_path ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt  \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--qrels_path  ./output/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000 \
--output_path ./output/dl-20-passage \
```

## 3. Fine-tuning LLaMA
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv \
--logging_steps 10 \
--per_device_train_batch_size 64 \
--num_epochs 5 \
--num_negs 1 
```

## 4. In-context learning using LLaMA

### Predicting the performance of BM25 on TREC-DL 19 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt  \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-19-passage \
```

### Predicting the performance of BM25 on TREC-DL 20 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-20-passage \ 
```

### Predicting the performance of BM25 on TREC-DL 21 
```bash
python judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v2-passage/queries/dl-21-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v2-passage/runs/dl-21-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v2-passage/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a \
--qrels_path ./datasets/msmarco-v2-passage/qrels/dl-21-passage.qrels.txt \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v2-passage/runs/dl-21-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-21-passage \
```

### Predicting the performance of BM25 on TREC-DL 22 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v2-passage/queries/dl-22-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v2-passage/runs/dl-22-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v2-passage/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a \
--qrels_path ./datasets/msmarco-v2-passage/qrels/dl-22-passage.qrels-withDupes.txt \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v2-passage/runs/dl-22-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-22-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-22-passage \
```

### Predicting the performance of ANCE on TREC-DL 19 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt  \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--qrels_path  ./output/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-19-passage \
```
### Predicting the performance of ANCE on TREC-DL 20 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--qrels_path  ./output/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-20-passage \
```

## 5. Evaluation

### Evaluate QPP effectiveness of QPP-GenRE (finetuned LLaMA) for predicting the performance of BM25 in terms of RR@10  
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-21-passage/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-21-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-22-passage/dl-22-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-22-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 
```

### Evaluate QPP effectiveness of QPP-GenRE (finetuned LLaMA) for predicting the performance of ANCE in terms of RR@10 
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric mrr@10

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric mrr@10
```

### Evaluate QPP effectiveness of QPP-GenRE (finetuned LLaMA)  for predicting the performance of BM25 in terms of nDCG@10 
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10

python -u evaluation.py \
--predicted_path ./output/dl-21-passage/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-21-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-22-passage/dl-22-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-22-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 
```

### Evaluate QPP effectiveness of QPP-GenRE (finetuned LLaMA)  for predicting the performance of ANCE in terms of nDCG@10
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric ndcg@10
```

### Evaluate QPP effectiveness of QPP-GenRE (in-context learning-based LLaMA) for predicting the performance of BM25 in terms of RR@10  
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-21-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-21-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 
```
### Evaluate QPP effectiveness of QPP-GenRE (in-context learning-based LLaMA) for predicting the performance of ANCE in terms of RR@10 
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric mrr@10

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric mrr@10
```

### Evaluate QPP effectiveness of QPP-GenRE (in-context learning-based LLaMA) for predicting the performance of BM25 in terms of nDCG@10 
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-21-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-21-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 
```

### Evaluate QPP effectiveness of QPP-GenRE (in-context learning-based LLaMA) for predicting the performance of ANCE in terms of nDCG@10
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric ndcg@10
```