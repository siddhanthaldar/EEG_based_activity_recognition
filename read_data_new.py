import os
import sys
from keras.models import load_model
import numpy as np

cq_order = ["F3", "FC5", "AF3", "F7", "T7",  "P7",  "O1",
            "O2", "P8",  "T8",  "F8", "AF4", "FC6", "F4",
            "F8", "AF4"]


WL = 150
fs = 128

col_list = ['F8', 'F3','FC5', 'AF3']
#cols = [cq_order.index(i) for i in col_list]
out_list = ['Nothing', 'Push', 'Lift', 'Drop', 'Disappear']
D = len(col_list)
no_of_outputs = len(out_list)

try:
    from emotiv import epoc, utils
except ImportError:
    sys.path.insert(0, "..")
    from emotiv import epoc, utils

def main():
    duration = float(WL/fs)

    
    channels = col_list
    print(channels)
    
    # Setup headset
    headset = epoc.EPOC(enable_gyro=False)
    if channels:
        headset.set_channel_mask(channels)

    # Acquire
    idx, data = headset.acquire_data_fast(duration)
    print(np.shape(data))
    #data = data[:, cols]
    data = np.transpose(data)
    data = data.reshape(1, data.shape[0], data.shape[1], 1)
    train_mean = np.load('train_mean.npy')
    train_std = np.load('train_std.npy')

    img_arr = (img_arr -train_mean.reshape((-1,D,WL,1)))/train_std.reshape((-1,D,WL,1))

    filepath = "./models/WL=150_labels_5_electrodes_['F8', 'F3', 'FC5', 'AF3']_lr_0.001_filter_no._[50, 50, 20]_filt_frac_img_[0.25, 0.25]_drop_0.1_one_d_200_epochs_200.h5"
    model = load_model(filepath)
    pred = model.predict(img_arr)
    print(pred)

    print ("Battery: %d %%" % headset.battery)
    print ("Contact qualities")
    print (headset.quality)

    #utils.save_as_matlab(data, headset.channel_mask)

    try:
        headset.disconnect()
    except e:
        print (e)

if __name__ == "__main__":
    sys.exit(main())