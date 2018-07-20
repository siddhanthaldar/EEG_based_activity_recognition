import pyedflib
import pandas as pd
import numpy as np
import os
WL = 150
for filename in os.listdir("./EDF_files/"):
	f = pyedflib.EdfReader("./EDF_files/" + filename)
	N = f.getNSamples()[0]
	N = int(N/WL)
	if 'push' in filename:
		labels = ['P' for i in range(N)]
	elif 'neutral' in filename:
		labels = ['N' for i in range(N)]
	elif 'lift' in filename:
		labels = ['L' for i in range(N)]
	elif 'drop' in filename:
		labels = ['D' for i in range(N)]
	elif 'diss' in filename:
		labels = ['Dis' for i in range(N)]

	f._close()
	del f


	out_list = ['Nothing', 'Push', 'Lift', 'Drop', 'Disappear']
	unique_label_list = ['N', 'P', 'L', 'D', 'Dis']
	one_hot_arr = np.zeros((len(labels), len(out_list)))
	for i in range(len(labels)):
		one_hot_arr[i][unique_label_list.index(labels[i])] = 1
	df = pd.DataFrame(one_hot_arr, columns = out_list)
	df.to_csv('./output_labels/sample_' + filename[:-4] + '_labels.csv', index = False)

