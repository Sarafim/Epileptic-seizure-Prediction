###################################################################################################
#   rewrite .mat into .csv
###################################################################################################
# patients=['patient_name']
# path_mat      = 'C:/Path_to_matfile/'
# path_csv      = 'C:/Path_to_new_generated_csvfile/'
# for patient in patients:
# 	item = load_mat(path_mat + patient + '.mat')
# 	item.to_csv(path_csv + patient + '.csv', mode = 'w',index=False)
###################################################################################################
import csv
import scipy.io as sio
import numpy as np
import pandas as pd

def load_mat(path):
	buff = sio.loadmat(path)
	buff = buff['d'][0,0]
	item = dict()
	labels = ['RR_2Hz','time_2Hz','seizureStart','seizureEnd','time','Fs','N','RR_raw','RR_pos','data']
	for l in labels:
		if l!='data':
			item[l]= np.array(buff[l].ravel())
		else:
			item[l] = np.array(buff[l][:,np.flatnonzero(buff['labels']==(['ECG ECG']))].ravel())
	return pd.DataFrame.from_dict(item, orient='index').transpose()

def load_column(path,data_type):
	csvfile=open(path)
	gen = csv.reader(csvfile, delimiter=',')
	row_first=gen.__next__()
	ind = row_first.index(data_type)
	item = dict()
	for row in gen:
		if '' == row[ind]:
			return	item
		else:
			try:
				item[row_first[ind]].append(float(row[ind]))
			except:
				item[row_first[ind]]=[float(row[ind])]
	return item
