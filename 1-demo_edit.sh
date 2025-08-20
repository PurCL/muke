#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python -m experiments.evaluate_uns \
    --alg_name=unke_Mat \
    --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=Qwen2.5-7B-Instruct-matryoshka.json \
    --ds_name=unke \
    --dataset_size_limit=10 \
    --num_edits=1