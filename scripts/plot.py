#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: plot.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Wed Dec  9 23:20:45 2020
# ************************************************************************/

import argparse
import numpy as np
import torch
import matplotlib
import pandas as pd


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--log-path", type=str, default="log", help="log file path")
	parser.add_argument("--csv-path", type=str, default="log", help="log file path")
	args = parser.parse_args()
	data = np.loadtxt(args.log_path, np.float64).T

	k_set = set(data[0])
	lambda_set = set(data[1])
	dataframe = pd.DataFrame({'k':data[0],'lambda':data[1], 'rmse':data[2]})
	dataframe.to_csv(args.csv_path,index=False,sep=',')


