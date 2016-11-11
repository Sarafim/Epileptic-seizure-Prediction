###################################################################################################
#   rewrite .mat into .csv
###################################################################################################
# item = load_mat('D:/Diploma/Signals/Row_Signals/' + test + '.mat')
# item.to_csv('D:/Diploma/Signals/Ready_Signals/' + test + '.csv', mode = 'w',index=False)
###################################################################################################
import csv
import scipy.io as sio
import numpy as np
import pandas as pd



def load_mat(path):
	buff = sio.loadmat(path)['d'][0,0]
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

def load_data(path,data_type):
	csvfile=open(path)
	gen = csv.reader(csvfile, delimiter=',')
	row_first=gen.__next__()
	ind = [row_first.index(i) for i in data_type]
	stop_flag = 0

	def prog_load(item, step_add,step_del, mode='all'):
		'''
		return  1 reach end of the file at this step
		return -1 end was reached before
		return  0 all is ok 
		'''
		nonlocal stop_flag

		count_step=1
		if stop_flag == 1 or stop_flag ==-1:
			stop_flag =-1
			return stop_flag
		
		for row in gen:
			for i in ind:
				if '' == row[i]:
					stop_flag = 1
					return stop_flag

				try:
					item[row_first[i]].append(float(row[i]))
				except:
					item[row_first[i]]=[float(row[i])]
			if mode != 'all':
				if step_add == count_step:
					if mode == 'step':
						for i in ind:
							del item[row_first[i]][:step_del]
					return stop_flag
				else:
					count_step+=1
		else:
			for i in ind:
				del item[row_first[i]][:step_del-1]
			stop_flag = 1
			return stop_flag
	return prog_load

def step_del(path,data_type,step_time=60):
	csvfile=open(path)
	gen = csv.reader(csvfile, delimiter=',')
	row_first=gen.__next__()
	ind = row_first.index(data_type)
	yield 0
	step=1
	count_time=step_time
	for row in gen:
		if row[ind]=='':
			yield step
		if float(row[ind])>count_time:
			yield step
			count_time+=step_time
			step=1
		else:
			step+=1
	else:
		return step	

def step_add(path,data_type,start_time=300,step_time=60):
	csvfile=open(path)
	gen = csv.reader(csvfile, delimiter=',')
	row_first=gen.__next__()
	ind = row_first.index(data_type)
	step=1
	for row in gen:
		if row[ind]=='':
			yield step
		if float(row[ind])>start_time:
			yield step
			start_time+=step_time
			step=1
		else:
			step+=1
	else:
		return step	