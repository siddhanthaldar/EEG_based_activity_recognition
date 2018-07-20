import numpy as np
import pandas as pd
import os

WL = 150

col_list = ['F8', 'F3','FC5', 'AF3']
out_list = ['Nothing', 'Push', 'Lift', 'Drop', 'Disappear']
D = len(col_list)
no_of_outputs = len(out_list)
#Reading data
#generalise window size, no. of outputs, no_of_channels etc.
req_arr_list = []
outputs_list = []
files_list = [file[:-4] for file in os.listdir('./EDF_files')]
for i in files_list:
	df = pd.read_csv('./csvs/sample_'+ i + '.csv', usecols = col_list)
	df2 = pd.read_csv('./output_labels/sample_' + i + '_labels.csv', usecols = out_list)
	shape=df2.shape[0]
	temp_outputs = df2.as_matrix()
	print(temp_outputs.shape)
	arr = np.transpose(df.as_matrix())
	print('no_of_sensors, no_of_samples')
	print(arr.shape)
	total_len = int(arr.shape[1]/WL) * WL
	print(total_len)
	temp = arr[:,0:total_len]
	req_arr_list.append(temp)
	outputs_list.append(temp_outputs)
req_arr = np.concatenate(tuple(req_arr_list), axis = 1)
outputs = np.concatenate(tuple(outputs_list), axis = 0)
img_list = []
out_list = []
for i in range(outputs.shape[0]):
	img_list.append(req_arr[:,i*WL:(i + 1) * WL])
	out_list.append(outputs[i,:])
	print(i)
img_arr = np.asarray(img_list).reshape((-1,D,WL,1))
target = np.asarray(out_list)
print('samples of target')
print(target[0:5,:])
print(img_arr.shape)
print(target.shape)
no_of_images = img_arr.shape[0]
from sklearn.model_selection import train_test_split
train_img, temp_img, train_output, temp_output = train_test_split(img_arr, target, test_size = 0.3, random_state = 1, shuffle = True, stratify = target)
valid_img, test_img, valid_output, test_output = train_test_split(temp_img, temp_output, test_size = 0.33, random_state = 1, shuffle = True, stratify = temp_output)

train_mean = np.mean(train_img, axis = 0)
train_std = np.std(train_img, axis = 0)
train_img = (train_img-train_mean.reshape((-1,D,WL,1)))/train_std.reshape((-1,D,WL,1))
valid_img = (valid_img-train_mean.reshape((-1,D,WL,1)))/train_std.reshape((-1,D,WL,1))
test_img = (test_img-train_mean.reshape((-1,D,WL,1)))/train_std.reshape((-1,D,WL,1))
print(train_img.shape, valid_img.shape, test_img.shape)
print(train_output.shape, valid_output.shape, test_output.shape)

#train_img is the training dataset, val_img is the validation dataset

from keras.layers import Conv2D,MaxPooling2D, Reshape, Dense, Dropout
from keras.engine.topology import Input
from keras.models import Model
#from keras.layers.extra import TimeDistributedConvolution2D as Time2D
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam

def train_model(model, filepath, log_filepath, train, output, valid, valid_output, epochs, batch_size):
    chkpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(log_filepath)
    model.fit(train, output, validation_data = (valid, valid_output), epochs=epochs, verbose=1, batch_size=batch_size, callbacks=[chkpt, csv_logger])
lr = 0.001
#factor = 0.9
#patience = 10
#min_lr = 0.0003
filt_frac_img = [0.25,0.25]
no_of_filters = [50, 50, 20] 
dropout = 0.1
one_d = 200
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor,
#                              patience=patience, min_lr=min_lr, verbose = 1)

#model = Sequential()
x =  Input(shape = (D,WL,1))
y = Conv2D(no_of_filters[0], (1,int(WL * filt_frac_img[0])))(x)
y = MaxPooling2D((1,2))(y)
y = Dropout(rate = dropout, seed = 1)(y)
y = Conv2D(no_of_filters[1],(1,int(WL * filt_frac_img[1])))(y)
y = MaxPooling2D((1,2))(y)
y = Dropout(rate = dropout, seed = 1)(y)
temp = Conv2D(no_of_filters[2],(1,y.get_shape().as_list()[2]))(y)
temp = Conv2D(one_d, (D,1))(temp)
temp = Reshape((-1,))(temp)
pred_out = Dense(no_of_outputs, activation = 'softmax')(temp)
model = Model(x, pred_out)
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = lr))
print(model.summary())
epochs = 200
batch_size = 10
training = 1
filename = 'WL=' + str(WL) + '_labels_' + str(no_of_outputs) + '_electrodes_' + str(col_list) + '_lr_' + str(lr) + '_filter_no._' + str(no_of_filters) +  '_filt_frac_img_'  + str(filt_frac_img) + '_drop_' + str(dropout) + '_one_d_' + str(one_d) + '_epochs_' + str(epochs)
with open('./arch_reports/' + filename + '_report.txt','w') as fh:
# Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))


filepath = './models/' + filename + '.h5'
log_filepath = './logs/' + filename + '_results.log'
if training == 1:
	print('Training starts')
	train_model(model, filepath, log_filepath, train_img, train_output, valid_img, valid_output, epochs, batch_size)

model = load_model(filepath)
train_pred = model.predict(train_img)
valid_pred = model.predict(valid_img)
test_pred = model.predict(test_img)
train_cnt = 0
valid_cnt = 0
test_cnt = 0
for i in range(train_pred.shape[0]):
	if np.argmax(train_pred[i,:]) == np.argmax(train_output[i,:]):
		train_cnt = train_cnt + 1

for i in range(valid_pred.shape[0]):
	if np.argmax(valid_pred[i,:]) == np.argmax(valid_output[i,:]):
		valid_cnt = valid_cnt + 1

for i in range(test_pred.shape[0]):
	if np.argmax(test_pred[i,:]) == np.argmax(test_output[i,:]):
		test_cnt = test_cnt + 1


print('Training acc.')
print(train_cnt/train_pred.shape[0])

print('Validation acc.')
print(valid_cnt/valid_pred.shape[0])

print('Testing acc.')
print(test_cnt/test_pred.shape[0])


with open('./arch_reports/' + filename + '_report.txt', 'a') as file:
    file.write('\n' + str(train_cnt/train_pred.shape[0]) + '\n' + str(valid_cnt/valid_pred.shape[0]) + '\n' + str(test_cnt/test_pred.shape[0]))

