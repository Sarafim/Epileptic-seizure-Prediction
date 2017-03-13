###################################################################################################
#   feature preprocessing
###################################################################################################
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
import csv

def interictal(patients,
			   path_features 	= 'D:/Diploma/Features/f_',
			   path_csv 		= 'D:/Diploma/Signals/',
			   preictal_size    = 300,
			   f_window_size 	= 300,
			   f_window_step 	= 60,
			   eig              = False,
			   eig_window_size 	= 4,
			   eig_num 			= 1):

	interictal_time    = list()
	interictal_feature = list()
	interictal_w       = list()
	interictal_v       = list()

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
						interictal_w.append(wn[eig_num].real)
						interictal_v.append(vn[:, eig_num].real)
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
		# interictal_v = np.array(interictal_v)
		# interictal_w = np.array(interictal_w)
		return interictal_w,interictal_v
	else:
		# interictal_time = np.array(interictal_time)
		# interictal_feature = np.array(interictal_feature)
		return interictal_time, interictal_feature

def preictal  (patients,
			   eig              = False,
			   path_features 	= 'D:/Diploma/Features/f_',
			   path_csv 		= 'D:/Diploma/Signals/',
			   f_window_size 	= 300,
			   f_window_step 	= 60,
			   eig_num 			= 1,
			   interval         = 1,
 			   eig_window_size 	= 4):

	preictal_time 	 = list()
	preictal_feature = list()
	preictal_w       = list()
	preictal_v       = list()

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
					preictal_w.append(wn[eig_num].real)
					preictal_v.append(vn[:, eig_num].real)
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
		# preictal_w = np.array(preictal_w)
		# preictal_v = np.array(preictal_v)
		return  preictal_w, preictal_v
	else:
		# preictal_time = np.array(preictal_time)
		# preictal_feature = np.array(preictal_feature)
		return preictal_time,preictal_feature

# def eig_separation(patients,
# 				   eig = True,
# 				   path_features = 'D:/Diploma/Features/f_',
# 				   path_csv = 'D:/Diploma/Signals/',
# 				   f_window_size = 300,
# 				   f_window_step = 60,
# 				   eig_window_size = 4,
# 				   eig_num = 1):
# 	interictal_time = list()
# 	interictal_feature = list()
# 	preictal_time = list()
# 	preictal_feature = list()
# 	interictal_w = list()
# 	interictal_v = list()
# 	preictal_w = list()
# 	preictal_v = list()
# 	for patient in patients:
# 		path_feature_csv = path_features + patient + '.csv'
# 		path_patient     = path_csv + patient + '.csv'

# 		patient_feature = pd.read_csv(path_feature_csv, header=0)
# 		try:
# 			seizure_start = load_column(path_patient, data_type='seizureStart')['seizureStart']
# 			seizure_stop  = load_column(path_patient, data_type='seizureEnd')['seizureEnd']
# 			seizure_start_time = seizure_start[0]
# 			seizure_stop_time  = seizure_stop[0]
# 		except:
# 			seizure_start_time = patient_feature.values[0, -1]
# 			seizure_stop_time  = patient_feature.values[0, -1]

# 		seizure_counter = 0
# 		break_counter = 0
# 		interictal_counter = 0
# 		preictal_counter   = 0
# 		for i in range(patient_feature.shape[1]):
# 			break_counter -=1
# 			if break_counter > 0:
# 				continue
# 			if(patient_feature.values[0, i] - patient_feature.values[0, i-1]) > f_window_size:
# 				break_counter = int(f_window_size/f_window_step)
# 				continue

# 			if (seizure_start_time - patient_feature.values[0, i]) > f_window_size :
# 				interictal_counter += 1
# 				preictal_counter    = 0
# 				interictal_time.append(patient_feature.values[0,i])
# 				interictal_feature.append(patient_feature.values[2:,i])
# 				if eig:
# 					if interictal_counter > (eig_window_size - 1):
# 						window_feature = np.array(interictal_feature[-eig_window_size:])
# 						covmat = np.cov(window_feature.T)
# 						wn, vn = np.linalg.eig(covmat)
# 						interictal_w.append(float(wn[eig_num]))
# 						interictal_v.append(vn[:, eig_num])

# 			else:
# 				if((seizure_start_time - patient_feature.values[0, i]) < f_window_size) & ((seizure_start_time - patient_feature.values[0, i]) > 0):
# 					interictal_counter = 0
# 					preictal_counter  += 1
# 					preictal_time.append(patient_feature.values[0,i])
# 					preictal_feature.append(patient_feature.values[2:,i])

# 					if eig:
# 						if preictal_counter > (eig_window_size - 1):
# 							window_feature = np.array(preictal_feature[-eig_window_size:])
# 							covmat = np.cov(window_feature.T)
# 							wn, vn = np.linalg.eig(covmat)
# 							preictal_w.append(float(wn[eig_num]))
# 							preictal_v.append(vn[:, eig_num])

# 				if(seizure_stop_time - patient_feature.values[0, i]) < -f_window_step:
# 					seizure_counter += 1

# 					if seizure_counter == seizure_start.__len__():
# 						seizure_start_time = patient_feature.values[0, -1] + 2*f_window_size
# 						seizure_stop_time  = patient_feature.values[0, -1] + 2*f_window_size
# 					else:
# 						seizure_start_time = seizure_start[seizure_counter]
# 						seizure_stop_time  = seizure_stop [seizure_counter]
# 	interictal_v = np.array(interictal_v)
# 	interictal_w = np.array(interictal_w)
# 	preictal_v   = np.array(preictal_v)
# 	preictal_w   = np.array(preictal_w)
# 	if eig:
# 		return interictal_w,interictal_v,preictal_w,preictal_v
# 	else:
# 		return interictal_time,interictal_feature,preictal_time,preictal_feature


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