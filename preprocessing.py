###################################################################################################
#   feature preprocessing
###################################################################################################
# pp.new_features( patient_test[0], start_time, step_time,features)
###################################################################################################
import numpy as np
import pandas as pd
import math as m
import load as l

def new_features( test, start_time, step_time,features):
	feature = preprocessing(path='D:/Diploma/Signals/Ready_Signals/' + test + '.csv',start_time=start_time,step_time=step_time,features = features)
	feature.to_csv('D:/Diploma/Signals/Features/f_' + test + '.csv', mode = 'w',index=False)
	writer = pd.ExcelWriter('D:/Diploma/Signals/Features/f_' + test + '.xlsx')
	feature.to_excel(writer,'Sheet1')
	writer.save()

def preprocessing(path, start_time, step_time,features):
	window = dict()
	window_Hz = dict()
	feature = dict()
	index=[]
	load = l.load_data(path,data_type=['RR_raw','RR_pos'])
	s_add = l.step_add(path,data_type='RR_pos',start_time=start_time,step_time=step_time)
	s_del = l.step_del(path,data_type='RR_pos',step_time=step_time)

	load_Hz = l.load_data(path,data_type=['RR_2Hz','time_2Hz'])
	s_add_Hz = l.step_add(path,data_type='time_2Hz',start_time=start_time,step_time=step_time)
	s_del_Hz = l.step_del(path,data_type='time_2Hz',step_time=step_time)

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















