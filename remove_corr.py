import numpy as np
import pandas as pd
import os

files_list = [file[:-4] for file in os.listdir('./EDF_files')]
df_list = []
for i in files_list:
	df_list.append(pd.read_csv('./csvs/sample_'+ i + '.csv', usecols = ['AF3', 'AF4', 'F3','F4','F7','F8', 'FC5', 'FC6', 'T7','T8', 'CQ_CMS','CQ_DRL','P7','P8','O1','O2']))

df = pd.concat(df_list, axis = 0)

print(df.columns)
print(df.shape)
df = df.loc[:, (df!=0).any(axis = 0)]

print(df.columns)
print(df.shape)
def correlation(df, thr):
	col_corr = set()
	corr_matrix = df.corr()
	for i in range(len(corr_matrix.columns)):
		for j in range(i):
			if corr_matrix.iloc[i,j] >= thr:
				colname = corr_matrix.columns[i]
				col_corr.add(colname)
				if colname in df.columns:
					del df[colname]
	print(col_corr)
	print(df.columns)
	print(df.shape)
correlation(df, 0.3)
col_n_zeros = [sum(df[col]!=0) for col in df.columns]
print(col_n_zeros)