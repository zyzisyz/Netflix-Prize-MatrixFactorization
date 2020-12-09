#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: data_preprocess.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Tue Dec  8 21:56:17 2020
# ************************************************************************/

import argparse
import numpy as np
from sklearn import preprocessing

def load_data(train_data_path, train_npy_save_path, test_data_path, test_npy_save_path):
	# load txt data file
	print("start to load train dataset from {}".format(train_data_path))
	train_data = np.loadtxt(train_data_path, dtype=np.str).T
	train_user = train_data[0]
	train_movie = train_data[1]
	train_score = train_data[2]
	print("train dataset ready")

	print("start to load test dataset from {}".format(test_data_path))
	test_data = np.loadtxt(test_data_path, dtype=np.str).T
	test_user = test_data[0]
	test_movie = test_data[1]
	test_score = test_data[2]
	print("test dataset ready")

	user_enc = preprocessing.LabelEncoder()
	movie_enc = preprocessing.LabelEncoder()

	user_enc.fit(train_user)
	movie_enc.fit(train_movie)

	print("transform user")
	train_user = user_enc.transform(train_user)
	test_user = user_enc.transform(test_user)

	print("transform movie")
	train_movie = movie_enc.transform(train_movie)
	test_movie = movie_enc.transform(test_movie)

	num_user = len(np.unique(train_user))
	num_movie = len(np.unique(train_movie))
	print("num_movie is: ", num_movie)
	print("num_user is: ", num_user)

	train_matrix = np.zeros((num_user, num_movie), dtype=np.float64)
	test_matrix = np.zeros((num_user, num_movie), dtype=np.float64)

	for idx in range(len(train_user)):
		user_index = train_user[idx]
		movie_index = train_movie[idx]
		train_matrix[user_index][movie_index] = float(int(train_score[idx]))

	for idx in range(len(test_user)):
		user_index = test_user[idx]
		movie_index = test_movie[idx]
		test_matrix[user_index][movie_index] = float(int(test_score[idx]))

	print("train matrix save to ", train_npy_save_path)
	np.save(train_npy_save_path, train_matrix)
	print("test matrix save to ", test_npy_save_path)
	np.save(test_npy_save_path, test_matrix)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train-data-load-path", type=str, default="", help="data file path")
	parser.add_argument("--test-data-load-path", type=str, default="", help="data file path")
	parser.add_argument("--train-npy-save-path", type=str, default="", help="npy file path")
	parser.add_argument("--test-npy-save-path", type=str, default="", help="npy file path")
	args = parser.parse_args()
	load_data(args.train_data_load_path, args.train_npy_save_path, args.test_data_load_path, args.test_npy_save_path)

