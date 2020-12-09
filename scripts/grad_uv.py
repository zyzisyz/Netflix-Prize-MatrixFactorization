#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: grad_uv.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Wed Dec  9 13:07:09 2020
# ************************************************************************/

import argparse
import numpy as np
import torch
import matplotlib
import tqdm

def grad_uv(train_matrix, test_matrix, k=25, lambda_parameters=0.1, device="cpu"):
	num_steps = 2000
	learning_rate = 0.00005
	N = 1719446

	A = (train_matrix>0).astype(train_matrix.dtype)
	mask = (test_matrix>0).astype(train_matrix.dtype)

	A = torch.from_numpy(A.copy()).to(device)
	mask = torch.from_numpy(mask.copy()).to(device)
	train_matrix = torch.from_numpy(train_matrix.copy()).to(device)
	test_matrix = torch.from_numpy(test_matrix.copy()).to(device)

	U = torch.rand((10000,k), dtype=torch.float32, device=device) * 0.001
	U.requires_grad = True
	V = torch.rand((k,10000), dtype=torch.float32, device=device) * 0.001
	V.requires_grad = True

	for step in range(num_steps):
		predict_matrix = torch.mm(U, V)
		J = 0.5*torch.norm(A.mul(train_matrix-predict_matrix)).pow(2) \
				+ lambda_parameters*torch.norm(U).pow(2) \
				+ lambda_parameters*torch.norm(V).pow(2)
		J.backward()
		rmse = torch.sqrt(torch.sum(torch.pow(test_matrix-torch.mul(mask, predict_matrix), 2))/N)

		with torch.no_grad():
			U -= U.grad*learning_rate
			V -= V.grad*learning_rate
			U.grad = None
			V.grad = None

		print("idx: {}\tRMSE: {:.2f}\tJ: {:.2f}".format(step, rmse.item(), J.item()))
		if step != 0:
			res = min(res, rmse)
		else:
			res = rmse
	return res


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train-matrix-path", type=str, default="", help="npy file path")
	parser.add_argument("--test-matrix-path", type=str, default="", help="npy file path")
	parser.add_argument("--log-path", type=str, default="log", help="log file path")
	args = parser.parse_args()

	train_matrix = np.load(args.train_matrix_path).astype(np.float32)
	test_matrix = np.load(args.test_matrix_path).astype(np.float32)

	with open(args.log_path, "w") as f:
		for k in [5, 10, 25, 30]:
			for lam in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.4]:
				rmse = grad_uv(train_matrix=train_matrix, test_matrix=test_matrix, k=k, lambda_parameters=lam, device="cuda")
				f.write("{} {} {}\n".format(k, lam, rmse))
			
