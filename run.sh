#!/bin/bash

#*************************************************************************
#	> File Name: run.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Tue Dec  8 21:46:50 2020
# ************************************************************************/

stage=1

if [ $stage -eq 1 ];then
	time python -u scripts/data_preprocess.py \
		--train-data-load-path data/netflix_train.txt  \
		--test-data-load-path data/netflix_test.txt  \
		--train-npy-save-path data/train_matrix.npy \
		--test-npy-save-path data/test_matrix.npy 
fi


if [ $stage -eq 2 ];then
	time python -u scripts/cf.py \
		--train-matrix-path data/train_matrix.npy \
		--test-matrix-path data/test_matrix.npy 
fi


if [ $stage -eq 3 ];then
	rm -rf log
	touch log
	time CUDA_VISIBLE_DEVICES=3 python -u scripts/grad_uv.py \
		--train-matrix-path data/train_matrix.npy \
		--test-matrix-path data/test_matrix.npy \
		--log-path log
fi


if [ $stage -eq 4 ];then
	time python -u scripts/plot.py \
		--log-path log \
		--csv-path result.csv 
fi
