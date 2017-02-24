from feature_selection      import new_features
# from feature_preproccesing  import interictal, preictal
import random
import csv
import numpy                 as np
import pandas                as pd
import sklearn.preprocessing as skpp
import sklearn.svm           as sksvm
import matplotlib.pyplot     as plt
import matplotlib.patches    as mpatches
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

def interictal(patients,
			   path_features 	= 'D:/Diploma/Features/f_',
			   path_csv 		= 'D:/Diploma/Signals/',
			   preictal_size    = 300,
			   f_window_size 	= 300,
			   f_window_step 	= 60,
			   eig              = False,
			   eig_window_size 	= 4):

	interictal_time    = list()
	interictal_feature = list()
	interictal_w       = list()
	interictal_v       = list()
	interictal_w_h  = list()
	interictal_v_h  = list()
	for i in range(8):
		interictal_w.append(list())
		interictal_v.append(list())
		interictal_w_h.append(list())
		interictal_v_h.append(list())

	for patient in patients:
		path_feature_csv = path_features + patient + '.csv'
		path_patient     = path_csv      + patient + '.csv'

		patient_feature = pd.read_csv(path_feature_csv, header=0)
		try:
			seizure_start = load_column(path_patient, data_type='seizureStart')['seizureStart']
			seizure_stop  = load_column(path_patient, data_type='seizureEnd')['seizureEnd']
			seizure_start_time = seizure_start[0]
			seizure_stop_time  = seizure_stop[0]
		except:
			seizure_start_time = patient_feature.values[0, -1]
			seizure_stop_time  = patient_feature.values[0, -1]

		seizure_counter = 0
		break_counter = 0
		interictal_counter = 0
		for i in range(patient_feature.shape[1]):
			break_counter -=1
			if break_counter > 0:
				continue
			if(patient_feature.values[0, i] - patient_feature.values[0, i-1]) > preictal_size:
				break_counter = int(f_window_size/f_window_step)
				continue

			if (seizure_start_time - patient_feature.values[0, i]) > preictal_size :
				interictal_counter += 1
				interictal_time.append(patient_feature.values[0,i])
				interictal_feature.append(patient_feature.values[2:,i])
				if eig:
					if interictal_counter > (eig_window_size - 1):
						window_feature = np.array(interictal_feature[-eig_window_size:])
						covmat = np.cov(window_feature.T)
						wn, vn = np.linalg.eig(covmat)
						ind = np.argsort(abs(wn))
						wn = wn[ind]
						vn = vn[:, ind]
						vn = vn.real
						vector = list(vn[:, 6])
						for j in range(8):
							vector.append(vn[j, 7])
						interictal_v[0].append(vector)
						interictal_w[0] = wn[6:7]
						# for eig_num in range(8):
						# 	interictal_w[eig_num].append(wn[eig_num].real)
						# 	interictal_v[eig_num].append(vn[:, eig_num].real)

						wn, vn = np.linalg.eigh(covmat)
						for eig_num in range(8):
							interictal_w_h[eig_num].append(wn[eig_num].real)
							interictal_v_h[eig_num].append(vn[:, eig_num].real)
			else:
				interictal_counter = 0
				if(seizure_stop_time - patient_feature.values[0, i]) < (-preictal_size + f_window_step):
					seizure_counter += 1
					if seizure_counter == seizure_start.__len__():
						seizure_start_time = patient_feature.values[0, -1] + 2*f_window_size
						seizure_stop_time  = patient_feature.values[0, -1] + 2*f_window_size
					else:
						seizure_start_time = seizure_start[seizure_counter]
						seizure_stop_time  = seizure_stop [seizure_counter]
	if eig:
		return interictal_w,interictal_v,interictal_w_h,interictal_v_h
	else:
		return interictal_time, interictal_feature

def preictal  (patients,
			   eig              = False,
			   path_features 	= 'D:/Diploma/Features/f_',
			   path_csv 		= 'D:/Diploma/Signals/',
			   f_window_size 	= 300,
			   f_window_step 	= 60,
			   interval         = 1,
 			   eig_window_size 	= 4):

	preictal_time 	 = list()
	preictal_feature = list()
	preictal_w       = list()
	preictal_v       = list()
	preictal_w_h     = list()
	preictal_v_h     = list()
	for i in range(8):
		preictal_w.append(list())
		preictal_v.append(list())
		preictal_w_h.append(list())
		preictal_v_h.append(list())
	for patient in patients:
		path_feature_csv = path_features + patient + '.csv'
		path_patient     = path_csv      + patient + '.csv'

		patient_feature  = pd.read_csv(path_feature_csv, header=0)
		try:
			seizure_start      = load_column(path_patient, data_type='seizureStart')['seizureStart']
			seizure_stop       = load_column(path_patient, data_type='seizureEnd')['seizureEnd']
			seizure_start_time = seizure_start[0]
			seizure_stop_time  = seizure_stop[0]
		except:
			seizure_start_time = patient_feature.values[0, -1]
			seizure_stop_time  = patient_feature.values[0, -1]

		seizure_counter = 0
		break_counter = 0
		for i in range(patient_feature.shape[1]):
			break_counter -=1
			if break_counter > 0:
				continue
			if(patient_feature.values[0, i] - patient_feature.values[0, i-1]) > f_window_size:
				break_counter = int(f_window_size/f_window_step)
				continue

			if ((seizure_start_time - patient_feature.values[0, i]) < f_window_step) & (
				(seizure_start_time - patient_feature.values[0, i]) > 0):
				if eig:
					for window in range(eig_window_size):
						new_index = int((eig_window_size - window - 1) * interval)
						preictal_time.append(patient_feature.values[0,  i-new_index])
						preictal_feature .append(patient_feature.values[2:, i-new_index])

					window_feature = np.array(preictal_feature[-eig_window_size:])
					covmat = np.cov(window_feature.T)
					wn, vn = np.linalg.eig(covmat)
					ind = np.argsort(abs(wn))
					wn = wn[ind]
					vn = vn[:, ind]
					vn = vn.real
					vector = list(vn[:, 6])
					for j in range(8):
						vector.append(vn[j, 7])
					preictal_v[0].append(vector)
					preictal_w[0] = wn[6:7]
					# for eig_num in range(8):
					# 	preictal_w[eig_num].append(wn[eig_num].real)
					# 	preictal_v[eig_num].append(vn[:,eig_num].real)
					# wn, vn = np.linalg.eigh(covmat)
					for eig_num in range(8):
						preictal_w_h[eig_num].append(wn[eig_num].real)
						preictal_v_h[eig_num].append(vn[:,eig_num].real)
				else:
					preictal_time.append(patient_feature.values[0, i])
					preictal_feature.append(patient_feature.values[2:, i])
			else:
				if (seizure_stop_time - patient_feature.values[0, i]) < (-f_window_size + f_window_step):
					seizure_counter += 1
					if seizure_counter == seizure_start.__len__():
						seizure_start_time = patient_feature.values[0, -1] + 2 * f_window_size
						seizure_stop_time  = patient_feature.values[0, -1] + 2 * f_window_size
					else:
						seizure_start_time = seizure_start[seizure_counter]
						seizure_stop_time  = seizure_stop[seizure_counter]

	if eig:
		return  preictal_w, preictal_v,preictal_w_h,preictal_v_h
	else:
		return preictal_time,preictal_feature

def model(interictal_w, preictal_w, interictal_w_h, preictal_w_h,interictal_v, preictal_v,interictal_v_h, preictal_v_h,train_data, train_part, path_result,sh):
	fig = plt.figure()
	ax = [0, 1, 2, 3, 4, 5, 6, 7]
	for eig_num in range(1):
		new_interictal = interictal_v[eig_num]
		new_preictal = preictal_v[eig_num]
		if sh is 1:
			arr_inter = [i for i in range(new_interictal.__len__())]
			arr_pre   = [i for i in range(new_preictal.__len__())]
			random.shuffle(arr_inter)
			random.shuffle(arr_pre)
			interictal = list()
			preictal   = list()
			for ind in arr_inter:
				interictal.append(new_interictal[arr_inter[ind]])
			for ind in arr_pre:
				preictal.append(new_preictal[arr_pre[ind]])
		else:
			interictal = new_interictal
			preictal   = new_preictal

		if train_data is 'interictal':
			X_train       = skpp.scale(interictal[:int(train_part*interictal.__len__())])
			X_test_train  = skpp.scale(interictal[int(train_part*interictal.__len__()):])
			X_test        = skpp.scale(preictal)
		else:
			X_train       = skpp.scale(preictal[:int(train_part*preictal.__len__())])
			X_test_train  = skpp.scale(preictal[int(train_part*preictal.__len__()):])
			X_test        = skpp.scale(interictal)
		nu_plot = list()
		predict_plot_X_test_train = list()
		predict_plot_X_test = list()
		predict_plot_success = list()
		for nu in range(99):
			nu = 0.01 * nu + 0.01
			clf = sksvm.OneClassSVM(nu=nu, kernel='rbf').fit(X_train)
			predict_X_test_train = clf.predict(X_test_train);
			predict_X_test       = clf.predict(X_test);
			prediction_success_X_test_train = 100 * (X_test_train.__len__() - sum([ 1 for i in predict_X_test_train if i < 0]))/X_test_train.__len__()
			prediction_success_X_test       = 100 * (X_test.__len__() - sum([ 1 for i in predict_X_test if i > 0]))/X_test.__len__()
			prediction_success              = (prediction_success_X_test_train + prediction_success_X_test)/2
			nu_plot.append(nu)
			predict_plot_X_test_train.append(prediction_success_X_test_train)
			predict_plot_X_test.append(prediction_success_X_test)
			predict_plot_success.append(prediction_success)

		red_patch = list()
		ax[eig_num] = fig.add_subplot(421 + eig_num)
		red_patch.append(mpatches.Patch(label='X_train.__len__ = ' + str(X_train.__len__())))
		red_patch.append(mpatches.Patch(label='X_test_train.__len__  = ' + str(X_test_train.__len__())))
		red_patch.append(mpatches.Patch(label='X_test.__len__  = ' + str(X_test.__len__())))
		red_patch.append(mpatches.Patch(label='X_test_train is blue'))
		red_patch.append(mpatches.Patch(label='X_test is green'))
		ax[eig_num].plot(nu_plot, predict_plot_X_test_train, 'b')
		ax[eig_num].plot(nu_plot, predict_plot_X_test, 'g')
		ax[eig_num].plot(nu_plot, predict_plot_success, 'r')

		ax[eig_num].title.set_text('Vector, eig num = '+str(eig_num))

		plt.legend(handles=red_patch, loc=4)
		ax[eig_num].set_ylabel('Prediction accuracy, %')
		ax[eig_num].set_xlabel('Fraction of training errors,%')
		ax[eig_num].grid(True)
	fig.set_size_inches(100, 50)
	if sh is 1:
		fig.savefig(path_result + '_eig_shuffled' +'.png')
	else:
		fig.savefig(path_result + '_eig' +'.png')
	plt.close(fig)

	fig = plt.figure()
	ax = [0, 1, 2, 3, 4, 5, 6, 7]
	for eig_num in range(8):
		new_interictal = interictal_v_h[eig_num]
		new_preictal = preictal_v_h[eig_num]
		if sh is 1:
			arr_inter = [i for i in range(new_interictal.__len__())]
			arr_pre   = [i for i in range(new_preictal.__len__())]
			random.shuffle(arr_inter)
			random.shuffle(arr_pre)
			interictal = list()
			preictal   = list()
			for ind in arr_inter:
				interictal.append(new_interictal[arr_inter[ind]])
			for ind in arr_pre:
				preictal.append(new_preictal[arr_pre[ind]])
		else:
			interictal = new_interictal
			preictal   = new_preictal

		if train_data is 'interictal':
			X_train       = skpp.scale(interictal[:int(train_part*interictal.__len__())])
			X_test_train  = skpp.scale(interictal[int(train_part*interictal.__len__()):])
			X_test        = skpp.scale(preictal)
		else:
			X_train       = skpp.scale(preictal[:int(train_part*preictal.__len__())])
			X_test_train  = skpp.scale(preictal[int(train_part*preictal.__len__()):])
			X_test        = skpp.scale(interictal)
		nu_plot = list()
		predict_plot_X_test_train = list()
		predict_plot_X_test = list()
		predict_plot_success = list()
		for nu in range(99):
			nu = 0.01 * nu + 0.01
			clf = sksvm.OneClassSVM(nu=nu, kernel='rbf').fit(X_train)
			predict_X_test_train = clf.predict(X_test_train);
			predict_X_test       = clf.predict(X_test);
			prediction_success_X_test_train = 100 * (X_test_train.__len__() - sum([ 1 for i in predict_X_test_train if i < 0]))/X_test_train.__len__()
			prediction_success_X_test       = 100 * (X_test.__len__() - sum([ 1 for i in predict_X_test if i > 0]))/X_test.__len__()
			prediction_success              = (prediction_success_X_test_train + prediction_success_X_test)/2
			nu_plot.append(nu)
			predict_plot_X_test_train.append(prediction_success_X_test_train)
			predict_plot_X_test.append(prediction_success_X_test)
			predict_plot_success.append(prediction_success)

		red_patch = list()
		ax[eig_num] = fig.add_subplot(421 + eig_num)
		red_patch.append(mpatches.Patch(label='X_train.__len__ = ' + str(X_train.__len__())))
		red_patch.append(mpatches.Patch(label='X_test_train.__len__  = ' + str(X_test_train.__len__())))
		red_patch.append(mpatches.Patch(label='X_test.__len__  = ' + str(X_test.__len__())))
		ax[eig_num].plot(nu_plot, predict_plot_X_test_train, 'b')
		ax[eig_num].plot(nu_plot, predict_plot_X_test, 'g')
		ax[eig_num].plot(nu_plot, predict_plot_success, 'r')

		ax[eig_num].title.set_text('Vector, eig num = '+str(eig_num))

		plt.legend(handles=red_patch, loc=4)
		ax[eig_num].set_ylabel('Prediction accuracy, %')
		ax[eig_num].set_xlabel('Fraction of training errors,%')
		ax[eig_num].grid(True)
	fig.set_size_inches(100, 50)
	if sh is 1:
		fig.savefig(path_result + '_eigh_shuffled' +'.png')
	else:
		fig.savefig(path_result + '_eigh' +'.png')
	plt.close(fig)

# window dict
window = dict()
window ['eig_size']      = []
window ['feature_size']  = []
window ['feature_step']  = []
window ['add_parameter'] = []
window ['feature_sh']    = []

# model settings
train_datas  = []
train_parts  = []

# HRV types
ecg_types = dict()
ecg_types['focal']       = 'Focal_Seizures_processed/'
ecg_types['generalized'] = 'Generalized_Seizures_processed/'

# path dict
path = dict()
path ['signal']  = 'D:/Diploma/Signals/'
path ['feature'] = 'D:/Diploma/exp_2/Features/'
path ['result']  = 'D:/Diploma/exp_2/Results/'

# features dict
features             = dict()
features['length']   = True
features['meanNN']   = True
features['SDNN']     = True
features['RMSSD']    = True
features['variance'] = True
features['NN50']     = True
features['FDF']      = True


# patient dict
patients = dict()
patients ['focal'] =        [
                            'boyko-18',
                            # 'bernatskaya1-5','bernatskaya-15',
                            # 'butkevych-15',
                            # 'gudz-5',
                            # 'kaganiuk-15',
                            # 'klavdiev-15',
                            # 'kozubal-13',
                            # 'lekhkodukh-15',
                            # 'mazanichka-15',
                            # 'nesterchuk1-14','nesterchuk2-15','nesterchuk3-15','nesterchuk4-15',
                            # 'podvizhenko-15',
                            # 'polulyah-14-15',
                            # 'reshetnik-15',
                            # 'serdich1-15','serdich-14',
                            # 'stupak-14',
                            # 'tarasov-18',
                            # 'tumareva-14-15',
                            # 'volkogon-15'
                            ]
patients ['generalized'] =  [
                            # 'drozdov-14',
                            # 'dukova-15',
                            # 'feschenko-15',
                            # 'karpenko-15',
                            # 'kharchenko-15-16',
                            # 'kostuk-15',
                            # 'levchenko1_26_01_14-15', 'levchenko2_21_06_14-14-15-16',
                            # 'marchenko1-15', 'marchenko-15',
                            # 'petrian-15',
                            # 'semashko1-14', 'semashko-14', 'semashko2-14',
                            # 'shamina-15',
                            # 'surdu-15',
                            # 'yakimets-14'
                            ]
############################################################################
#                               Test 1                                     #
############################################################################
window ['eig_size'].append(2)
window ['feature_size'].append(300)
window ['feature_step'].append({'interictal' : 60,'preictal' : 10})
window ['add_parameter'].append({'preictal_size' : 360 ,'interval' : 6})
window ['feature_sh'].append('only_1')
train_datas.append('interictal')
train_parts.append(0.8)
############################################################################
#                               Test 2                                     #
############################################################################
window ['eig_size'].append(2)
window ['feature_size'].append(300)
window ['feature_step'].append({'interictal' : 60,'preictal' : 10})
window ['add_parameter'].append({'preictal_size' : 360 ,'interval' : 6})
window ['feature_sh'].append('only_1')
train_datas.append('preictal')
train_parts.append(0.8)
###########################################################################
#                               Test 3                                    #
###########################################################################
window ['eig_size'].append(2)
window ['feature_size'].append(660)
window ['feature_step'].append({'interictal' : 60,'preictal' : 10})
window ['add_parameter'].append({'preictal_size' : 720 ,'interval' : 6})
window ['feature_sh'].append('only_1')
train_datas.append('interictal')
train_parts.append(0.8)
############################################################################
#                               Test 4                                     #
############################################################################
window ['eig_size'].append(2)
window ['feature_size'].append(660)
window ['feature_step'].append({'interictal' : 60,'preictal' : 10})
window ['add_parameter'].append({'preictal_size' : 720 ,'interval' : 6})
window ['feature_sh'].append('only_1')
train_datas.append('preictal')
train_parts.append(0.8)
###########################################################################
#                               Test 5                                    #
###########################################################################
window ['eig_size'].append(2)
window ['feature_size'].append(920)
window ['feature_step'].append({'interictal' : 60,'preictal' : 10})
window ['add_parameter'].append({'preictal_size' : 980 ,'interval' : 6})
window ['feature_sh'].append('only_1')
train_datas.append('interictal')
train_parts.append(0.8)
############################################################################
#                               Test 6                                     #
############################################################################
window ['eig_size'].append(2)
window ['feature_size'].append(920)
window ['feature_step'].append({'interictal' : 60,'preictal' : 10})
window ['add_parameter'].append({'preictal_size' : 780 ,'interval' : 6})
window ['feature_sh'].append('only_1')
train_datas.append('preictal')
train_parts.append(0.8)
############################################################################
#                   Feature selections                                     #
############################################################################
print('Feature selections')
path_feature_csv = dict()
path_feature_xlsx = dict()
for ecg_type in ecg_types:
	for patient in patients[ecg_type]:
		path_source = path['signal'] + ecg_types[ecg_type] + patient + '.csv'
		for i in range(window['feature_size'].__len__()):
			if i in [0,2,4]:
				feature_size = window['feature_size'][i]
				feature_step = window['feature_step'][i]

				path_feature_csv['interictal']  = path['feature']+ ecg_types[ecg_type] + 'interictal_csv/'   + str(feature_size) + '_' + str(feature_step['interictal']) + '_' + patient + '.csv'
				path_feature_xlsx['interictal'] = path['feature']+ ecg_types[ecg_type] + 'interictal_excel/' + str(feature_size) + '_' + str(feature_step['interictal']) + '_' + patient + '.xlsx'
				path_feature_csv['preictal']    = path['feature']+ ecg_types[ecg_type] + 'preictal_csv/'     + str(feature_size) + '_' + str(feature_step['preictal']) + '_' + patient + '.csv'
				path_feature_xlsx['preictal']   = path['feature']+ ecg_types[ecg_type] + 'preictal_excel/'   + str(feature_size) + '_' + str(feature_step['preictal']) + '_' + patient + '.xlsx'

				print('interictal: ' + patient + '   ' + str(feature_size) + '   ' + str(feature_step['interictal']))
				new_features(path_source,path_feature_csv['interictal'],path_feature_xlsx['interictal'], feature_size,feature_step['interictal'],features)

				print('preictal  : ' + patient + '   ' + str(feature_size) + '   ' + str(feature_step['preictal']))
				new_features(path_source,path_feature_csv['preictal'],path_feature_xlsx['preictal'], feature_size,feature_step['preictal'],features)
############################################################################
#                   Model                                                  #
############################################################################
path_features  = dict()
path_sources   = dict()
for i in range(window['feature_size'].__len__()):
	eig_size       = window['eig_size'][i]
	add_parameter  = window['add_parameter'][i]
	feature_size   = window['feature_size'][i]
	feature_step   = window['feature_step'][i]
	feature_sh     = window['feature_sh'][i]
	for ecg_type in ecg_types:
		path_features['interictal']  = path['feature']+ ecg_types[ecg_type] + 'interictal_csv/'   + str(feature_size) + '_' + str(feature_step['interictal']) + '_'
		path_features['preictal']    = path['feature']+ ecg_types[ecg_type] + 'preictal_csv/'     + str(feature_size) + '_' + str(feature_step['preictal']) + '_'
		path_sources                 = path['signal'] + ecg_types[ecg_type]

		interictal_w, interictal_v,	interictal_w_h, interictal_v_h,  = interictal(  patients[ecg_type],
																															eig             = True,
																															path_features   = path_features['interictal'],
																															path_csv        = path_sources,
																															preictal_size   = add_parameter['preictal_size'],
																															f_window_size   = feature_size,
																															f_window_step   = feature_step['interictal'],
																															eig_window_size = eig_size
																														)
		preictal_w,   preictal_v, 	preictal_w_h,   preictal_v_h   = preictal  (   patients[ecg_type],
																															eig             = True,
																															path_features   = path_features['preictal'],
																															path_csv        = path_sources,
																															interval        = add_parameter['interval'],
																															f_window_size   = feature_size,
																															f_window_step   = feature_step['preictal'],
																															eig_window_size = eig_size
																														 )
		train_data = train_datas[i]
		train_part = train_parts[i]
		path_result = path['result']   + train_data +'_'+ str(train_part) + '_' + str(feature_size) + '_' + str(feature_step['interictal']) + '/' +  ecg_type + '_2features'
		if feature_sh is 'only_0':
			model(list(interictal_w), list(preictal_w), list(interictal_w_h), list(preictal_w_h),list(interictal_v), list(preictal_v),list(interictal_v_h), list(preictal_v_h),train_data, train_part, str(path_result),0)
		if feature_sh is 'only_1':
			model(list(interictal_w), list(preictal_w), list(interictal_w_h), list(preictal_w_h),list(interictal_v), list(preictal_v),list(interictal_v_h), list(preictal_v_h),train_data, train_part, str(path_result),1)
		if feature_sh is 'both':
			model(list(interictal_w), list(preictal_w), list(interictal_w_h), list(preictal_w_h), list(interictal_v),list(preictal_v), list(interictal_v_h), list(preictal_v_h), train_data, train_part, str(path_result),0)
			model(list(interictal_w), list(preictal_w), list(interictal_w_h), list(preictal_w_h), list(interictal_v),list(preictal_v), list(interictal_v_h), list(preictal_v_h), train_data, train_part, str(path_result),1)

		print('TEST FINISHED')


