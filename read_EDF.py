import pyedflib
import pandas as pd
import numpy as np
import os
for filename in os.listdir("./EDF_files/"):
	f = pyedflib.EdfReader("./EDF_files/" + filename)
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	#sigbufs = np.zeros((n, f.getNSamples()[0]))
	df = pd.DataFrame(columns = signal_labels)
	for i in np.arange(n):
		df[signal_labels[i]] = f.readSignal(i).tolist()
	df.to_csv('./csvs/sample_' + filename[:-4] + '.csv', index = False)
	f._close()
	del f
