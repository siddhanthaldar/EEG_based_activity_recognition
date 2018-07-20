import numpy as np
import pandas as pd
import os
from keras.models import load_model
WL = 150

col_list = ['F8', 'F3', 'AF3']
out_list = ['Nothing', 'Push', 'Lift', 'Drop']
D = len(col_list)
no_of_outputs = len(out_list)
#Reading data
#generalise window size, no. of outputs, no_of_channels etc.
req_arr_list = []
outputs_list = []
files_list = [file[:-4] for file in os.listdir('./EDF_files')]
for i in files_list:
	df = pd.read_csv('./csvs/sample_'+ i + '.csv', usecols = col_list)
	arr = np.transpose(df.as_matrix())
	print('no_of_sensors, no_of_samples')
	print(arr.shape)
	total_len = int(arr.shape[1]/WL) * WL
	print(total_len)
	temp = arr[:,0:total_len]
	req_arr_list.append(temp)
	
req_arr = np.concatenate(tuple(req_arr_list), axis = 1)

img_list = []

no_of_windows = int(req_arr.shape[1]/WL)
print(no_of_windows)
for i in range(no_of_windows):
	img_list.append(req_arr[:,i*WL:(i + 1) * WL])
	
	print(i)
img_arr = np.asarray(img_list).reshape((-1,D,WL,1))


print(img_arr.shape)

no_of_images = img_arr.shape[0]

train_mean = np.load('train_mean.npy')
train_std = np.load('train_std.npy')

img_arr = (img_arr -train_mean.reshape((-1,D,WL,1)))/train_std.reshape((-1,D,WL,1))

filepath = "./models/WL=150_files_['ktz_lift-2', 'ktz_push-2', 'ktz_lift-1', 'ktz_push-1', 'ktz_drop-1', 'ktz_drop-2', 'ktz_neutral-1']_electrodes_['F8', 'F3', 'AF3']_lr_0.001_filter_no._[50, 50, 20]_filt_frac_img_[0.25, 0.25]_drop_0.2_one_d_400_epochs_400.h5"
model = load_model(filepath)
pred = model.predict(img_arr)
print(pred)



