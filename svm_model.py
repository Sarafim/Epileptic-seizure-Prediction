###################################################################################################
#   SVM model
###################################################################################################
# clf, threshold = svm_ml.train_model(patient_model,outlier_fraction,kernel)
# svm_ml.test_model(clf, threshold,patient_test,sel_max,sel_min)
###################################################################################################
import load as l
import numpy as np
import pandas as pd
import scipy as sc
import sklearn.svm as sksvm
import sklearn.preprocessing as skpr
import sklearn.decomposition as skdec

import matplotlib.pyplot as plt
import matplotlib.font_manager

def prepare_to_model(path):
	patient = pd.read_csv(path, header=0)

	patient_params = np.array(patient.values[:,1:])
	patient_params = skpr.scale(patient_params)

	return skdec.PCA(n_components=2).fit_transform(patient_params)

def sel_answer(max,min,data):
	start = data[0:max]
	is_in=np.ones(max)
	g=0
	for i in data[max:]:
		i = (i == 1)
		start = np.append(i,start)
		start = np.delete(start,max)
		g+=1
		if sum(start) <= max/min:
			is_in = np.append([0], is_in)
		else:
			is_in = np.append([1], is_in)

	return is_in[::-1]

def train_model(patient_model, kernel):
	for model in patient_model:
		try:
			X = np.concatenate((X, prepare_to_model('D:/Diploma/Signals/Features/f_' + model + '.csv')), axis=0)
		except:
			X = prepare_to_model('D:/Diploma/Signals/Features/f_' + model + '.csv')

	return  sksvm.OneClassSVM(kernel=kernel).fit(X)

def test_model(clf, patient_test,max,min,percent,predict_mode):
	for test in patient_test:
		patient = pd.read_csv('D:/Diploma/Signals/Features/f_' + test + '.csv', header=0)

		Y = prepare_to_model('D:/Diploma/Signals/Features/f_' + test + '.csv')
		dist_to_border = clf.decision_function(Y).ravel()
		if predict_mode:
			data = clf.predict(Y)
		else:
			data = dist_to_border > sc.stats.scoreatpercentile(dist_to_border,100 * percent)
		is_in = sel_answer(max,min, data)

		seizureStart = l.load_column(path='D:/Diploma/Signals/Ready_Signals/' + test + '.csv', data_type = 'seizureStart')
		seizureEnd = l.load_column(path='D:/Diploma/Signals/Ready_Signals/' + test + '.csv', data_type = 'seizureEnd')
		chance = clf.decision_function(Y).ravel()

		patient_r = pd.DataFrame(patient[is_in == 0]['0:f_time'])
		try:
			patient_r['seizureStart'] = pd.Series(np.array(seizureStart['seizureStart']),index=patient_r.index[:len(seizureStart['seizureStart'])])
			patient_r['seizureEnd'] = pd.Series(np.array(seizureEnd['seizureEnd']),index=patient_r.index[:len(seizureStart['seizureStart'])])
		except:
			pass
		patient_r['chance'] = pd.Series(np.array(chance[is_in == 0]),index=patient_r.index)

		writer = pd.ExcelWriter('D:/Diploma/Signals/Result/' + test + '_result.xlsx')
		patient_r.to_excel(writer,'Sheet1')
		writer.save()

def plot_2model(clf,patient_test,sel_min,sel_max,start_time,step_time,percent,predict_mode):
	for test in patient_test:
		fig = plt.figure()
		xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)

		for test in patient_test:
			try:
				Y = np.concatenate((Y, prepare_to_model('D:/Diploma/Signals/Features/f_' + test + '.csv')), axis=0)
			except:
				Y = prepare_to_model('D:/Diploma/Signals/Features/f_' + test + '.csv')

		if predict_mode:
			threshold = 0
			data = clf.predict(Y)
		else:
			dist_to_border = clf.decision_function(Y).ravel()
			threshold = sc.stats.scoreatpercentile(dist_to_border, 100 * percent)
			data = dist_to_border > threshold

		plt.title("Outlier detection")
		plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
		a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
		plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')

		seizureStart = l.load_column(path='D:/Diploma/Signals/Ready_Signals/' + test + '.csv', data_type='seizureStart')
		real_seizure = list()
		j=0
		color = [0]

		f_time = l.load_column(path='D:/Diploma/Signals/Features/f_' + test + '.csv', data_type='0:f_time')
		for i in f_time['0:f_time']:
			if j < len(seizureStart['seizureStart']):
				if (seizureStart['seizureStart'][j]-1200)<i:
					real_seizure.append(1)
					color.append(color[-1]+1)
					if (seizureStart['seizureStart'][j]) < i:
						j+=1
				else:
					real_seizure.append(0)
			else:
				real_seizure.append(0)
		real_seizure = np.array(real_seizure)
		color = np.array(color[0:-1])


		predict = sel_answer(sel_max,sel_min,data)
		b = plt.scatter(Y[predict == 0, 0], Y[predict == 0, 1], c='black')
		c = plt.scatter(Y[predict == 1, 0], Y[predict == 1, 1], c='white')
		d = plt.scatter(Y[real_seizure == 1, 0], Y[real_seizure == 1, 1], c=color)

		plt.axis('tight')
		plt.legend([a.collections[0], b, c,d],['learned decision function', 'is_out', 'is_in', 'real_seizure'],prop=matplotlib.font_manager.FontProperties(size=11))
		plt.xlim((-7, 7))
		plt.ylim((-7, 7))
		plt.show()
		fig.savefig('D:/Diploma/Test_graph_model/' + test + '_' + str(start_time) + '_' + str(step_time) + '.png')  # save the figure to file
		plt.close(fig)
#