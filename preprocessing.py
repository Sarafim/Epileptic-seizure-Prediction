###################################################################################################
#   feature preprocessing
###################################################################################################
# pp.new_features( patient_test[0], f_window_size, f_window_step,features)
# interictal_w,interictal_v,preictal_w,preictal_v = pp.eig_separation(patients,
#                                                                     eig = True,
#                                                                     path_features = 'D:/Diploma/Features/f_',
#                                                                     path_csv = 'D:/Diploma/Signals/',
#                                                                     f_window_size = 300,
#                                                                     f_window_step = 60,
#                                                                     eig_window_size = 4,
#                                                                     eig_num = 1)
###################################################################################################
import numpy as np
import pandas as pd
import math as m
import load as ld

def new_features( path_source, path_feature_csv,path_feature_xlsx, f_window_size, f_window_step,features):
	feature = preprocessing(path_source,f_window_size=f_window_size,f_window_step=f_window_step,features = features)
	feature = feature.T
	feature.to_csv(path_feature_csv, mode = 'w',index=False)
	writer = pd.ExcelWriter(path_feature_xlsx)
	feature.to_excel(writer,'Sheet1')
	writer.save()

def preprocessing(path, f_window_size, f_window_step,features):
	window = dict()
	window_Hz = dict()
	feature = dict()
	index=[]
	load = ld.load_data(path,data_type=['RR_raw','RR_pos'])
	s_add = ld.step_add(path,data_type='RR_pos',f_window_size=f_window_size,f_window_step=f_window_step)
	s_del = ld.step_del(path,data_type='RR_pos',f_window_step=f_window_step)

	load_Hz = ld.load_data(path,data_type=['RR_2Hz','time_2Hz'])
	s_add_Hz = ld.step_add(path,data_type='time_2Hz',f_window_size=f_window_size,f_window_step=f_window_step)
	s_del_Hz = ld.step_del(path,data_type='time_2Hz',f_window_step=f_window_step)

	load(item = window, step_add=s_add.__next__(),step_del=s_del.__next__(),mode='first')
	load_Hz(item = window_Hz, step_add=s_add_Hz.__next__(),step_del=s_del_Hz.__next__(),mode='first')

	while True:
		if len(window['RR_raw'])>5:
			feature_ext(item = feature,index=index,window = window,window_Hz = window_Hz,features = features)
		flag=load(item = window,step_add=s_add.__next__(), step_del=s_del.__next__(),mode='step')
		flag_Hz=load_Hz(item = window_Hz, step_add=s_add_Hz.__next__(),step_del=s_del_Hz.__next__(),mode='step')
		if(flag==1 | flag_Hz ==1):
			break
	
	return pd.DataFrame(feature,index=index)

def feature_ext(item,index,window,window_Hz,features):
	arrays={i:np.array(window[i]) for i in window}
	arrays_Hz={i:np.array(window_Hz[i]) for i in window_Hz}

	dataframe_index(index,arrays)
	final_time(item,arrays)
	if(features['length']):feture_len(item,arrays)
	if(features['meanNN']):feature_meanNN(item,arrays)
	if(features['SDNN']):feature_SDNN(item,arrays)
	if(features['RMSSD']):feature_RMSSD(item,arrays)
	if(features['variance']):feature_variance(item,arrays)
	if(features['NN50']):feature_NN50(item,arrays)
	if(features['FDF']):feature_FDF(item,arrays_Hz)


def dataframe_index(index,arrays):
	try:
		index.append(str(int(arrays['RR_pos'][0]))+':'+str(int(arrays['RR_pos'][-1])))
	except:
		index=[str(int(arrays['RR_pos'][0]))+':'+str(int(arrays['RR_pos'][-1]))]

def final_time(item,arrays):
	try:
		item['0:f_time'].append(str(int(arrays['RR_pos'][-1])))
	except:
		item['0:f_time']=[str(int(arrays['RR_pos'][-1]))]

def feture_len(item,arrays):
	try:
		item['1:length'].append(arrays['RR_raw'].size)
	except:
		item['1:length']=[arrays['RR_raw'].size]

def feature_meanNN(item,arrays):	
	try:
		item['2:meanNN'].append(arrays['RR_raw'].mean())
	except:
		item['2:meanNN']=[arrays['RR_raw'].mean()]

def feature_SDNN(item,arrays):	
	try:
		item['3:SDNN'].append(arrays['RR_raw'].std())
	except:
		item['3:SDNN']=[arrays['RR_raw'].std()]

def feature_RMSSD(item,arrays):
	buf=np.array([(arrays['RR_raw'][i]-arrays['RR_raw'][i-1])**2	for i in range(1,len(arrays))])
	try:
		item['4:RMSSD'].append(m.sqrt(buf.mean()))
	except:
		item['4:RMSSD']=[m.sqrt(buf.mean())]

def feature_variance(item,arrays):
	try:
		item['5:variance'].append(arrays['RR_raw'].var())
	except:
		item['5:variance']=[arrays['RR_raw'].var()]	

def feature_NN50(item,arrays):
	buf=[arrays['RR_raw'][i]-arrays['RR_raw'][i-1]	for i in range(1,len(arrays['RR_raw']))]
	buf=np.array([i for i in buf if i>0.05])
	try:
		item['6:NN50'].append(buf.size)
	except:
		item['6:NN50']=[buf.size]

def feature_FDF(item,arrays):
	Fs = 2
	N=len(arrays['RR_2Hz'])
	window = np.hanning(N)

	spectrum = np.abs(np.fft.rfft(window*arrays['RR_2Hz']))
	f=np.fft.rfftfreq(N, 1./Fs)

	powerLF = [spectrum[list(f).index(i)]*spectrum[list(f).index(i)] for i in f if (i > 0.04) & (i< 0.15)]			
	powerHF = [spectrum[list(f).index(i)]*spectrum[list(f).index(i)] for i in f if (i > 0.15) & (i< 0.4)]			
	try: 
		item['LF'].append(sum(powerLF)/len(powerLF))
		item['HF'].append(sum(powerHF)/len(powerHF))
		item['LF/HF'].append(item['LF'][-1]/item['HF'][-1])
	except:
		item['LF']=[sum(powerLF)/len(powerLF)]
		item['HF']=[sum(powerHF)/len(powerHF)]
		item['LF/HF']=[item['LF'][-1]/item['HF'][-1]]

def eig_separation(patients,
				   eig = True,
				   path_features = 'D:/Diploma/Features/f_',
				   path_csv = 'D:/Diploma/Signals/',
				   f_window_size = 300,
				   f_window_step = 60,
				   eig_window_size = 4,
				   eig_num = 1):
	interictal_time = list()
	interictal_feature = list()
	preictal_time = list()
	preictal_feature = list()
	interictal_w = list()
	interictal_v = list()
	preictal_w = list()
	preictal_v = list()
	for patient in patients:
		path_feature_csv = path_features + patient + '.csv'
		path_patient     = path_csv + patient + '.csv'

		patient_feature = pd.read_csv(path_feature_csv, header=0)
		try:
			seizure_start = ld.load_column(path_patient, data_type='seizureStart')['seizureStart']
			seizure_stop  = ld.load_column(path_patient, data_type='seizureEnd')['seizureEnd']
			seizure_start_time = seizure_start[0]
			seizure_stop_time  = seizure_stop[0]
		except:
			seizure_start_time = patient_feature.values[0, -1]
			seizure_stop_time  = patient_feature.values[0, -1]

		seizure_counter = 0
		break_counter = 0
		interictal_counter = 0
		preictal_counter   = 0
		for i in range(patient_feature.shape[1]):
			break_counter -=1
			if break_counter > 0:
				continue
			if(patient_feature.values[0, i] - patient_feature.values[0, i-1]) > f_window_size:
				break_counter = int(f_window_size/f_window_step)
				continue

			if (seizure_start_time - patient_feature.values[0, i]) > f_window_size :
				interictal_counter += 1
				preictal_counter    = 0
				interictal_time.append(patient_feature.values[0,i])
				interictal_feature.append(patient_feature.values[2:,i])
				if eig:
					if interictal_counter > (eig_window_size - 1):
						window_feature = np.array(interictal_feature[-eig_window_size:])
						covmat = np.cov(window_feature.T)
						wn, vn = np.linalg.eig(covmat)
						interictal_w.append(float(wn[eig_num]))
						interictal_v.append(vn[:, eig_num])

			else:
				if((seizure_start_time - patient_feature.values[0, i]) < f_window_size) & ((seizure_start_time - patient_feature.values[0, i]) > 0):
					interictal_counter = 0
					preictal_counter  += 1
					preictal_time.append(patient_feature.values[0,i])
					preictal_feature.append(patient_feature.values[2:,i])

					if eig:
						if preictal_counter > (eig_window_size - 1):
							window_feature = np.array(preictal_feature[-eig_window_size:])
							covmat = np.cov(window_feature.T)
							wn, vn = np.linalg.eig(covmat)
							preictal_w.append(float(wn[eig_num]))
							preictal_v.append(vn[:, eig_num])

				if(seizure_stop_time - patient_feature.values[0, i]) < -f_window_step:
					seizure_counter += 1

					if seizure_counter == seizure_start.__len__():
						seizure_start_time = patient_feature.values[0, -1] + 2*f_window_size
						seizure_stop_time  = patient_feature.values[0, -1] + 2*f_window_size
					else:
						seizure_start_time = seizure_start[seizure_counter]
						seizure_stop_time  = seizure_stop [seizure_counter]
	interictal_v = np.array(interictal_v)
	interictal_w = np.array(interictal_w)
	preictal_v   = np.array(preictal_v)
	preictal_w   = np.array(preictal_w)
	if eig:
		return interictal_w,interictal_v,preictal_w,preictal_v
	else:
		return interictal_time,interictal_feature,preictal_time,preictal_feature














