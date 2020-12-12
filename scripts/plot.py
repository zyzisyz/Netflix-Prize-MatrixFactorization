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
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--log-path", type=str, default="log", help="log file path")
	parser.add_argument("--csv-path", type=str, default="test.csv", help="log file path")
	parser.add_argument("--png-path", type=str, default="test.png", help="log file path")
	args = parser.parse_args()
	data = np.loadtxt(args.log_path, np.float64).T

	ks = data[0]
	lambs = data[1]
	rmses = data[2]

	ktype = ks[0]
	x = []
	y = []
	for i, k in enumerate(ks):
		print(k)
		if ktype != k:
			print(k)
			plt.scatter(x, y)
			plt.plot(x, y, label="k = "+str(int(k)))
			ktype = k
			x = []
			y = []
		x.append(lambs[i])
		y.append(rmses[i])

	plt.legend()
	plt.ylabel("RMSE")
	plt.xlabel("lambda")

	plt.grid()
	plt.show()

	if args.png_path is not None:
		plt.savefig(args.png_path)

	if args.csv_path is not None:
		dataframe = pd.DataFrame({'k':data[0],'lambda':data[1], 'rmse':data[2]})
		dataframe.to_csv(args.csv_path,index=False,sep=',')

	plt.close()
