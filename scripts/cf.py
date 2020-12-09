#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: cf.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Tue Dec  8 22:46:41 2020
# ************************************************************************/

import argparse
import numpy as np
from numpy.linalg import multi_dot
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

def Collaborative_Filtering(train_matrix_path, test_marix_path, N):
	# load data
	print("loading data")
	train_matrix = np.load(train_matrix_path)
	test_matrix = np.load(test_marix_path)

	print(train_matrix.shape)
	print(test_matrix.shape)

	# cosine similarity
	print("cosine_similarity in train matrix")
	train_matrix_cos = cosine_similarity(train_matrix)
	
	# CF predict matrix
	print("cf matrix predicting...")

	cf_predict = np.dot(train_matrix_cos, train_matrix)/np.dot(train_matrix, np.ones_like(train_matrix,dtype=np.float64))

	# root mean squared error
	mask = (test_matrix>0).astype(cf_predict.dtype)
	mask = np.multiply(cf_predict, mask)
	rmse = np.sqrt(np.sum(np.power(test_matrix-mask, 2))/N)
	return rmse

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--train-matrix-path", type=str, default="", help="npy file path")
	parser.add_argument("--test-matrix-path", type=str, default="", help="npy file path")
	args = parser.parse_args()

	rmse = Collaborative_Filtering(args.train_matrix_path, args.test_matrix_path, 1719446)

	print("Root Mean Squared Eroor: ", rmse)

