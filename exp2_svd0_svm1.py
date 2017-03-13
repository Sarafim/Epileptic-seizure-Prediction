import numpy                 as np
import pandas                as pd
import math                  as m
import sklearn.preprocessing as skpp
import sklearn.svm           as sksvm
import matplotlib.pyplot     as plt
import matplotlib.patches    as mpatches
import csv
import random
import os

def new_features(path_source, path_feature_csv, path_feature_xlsx, f_window_size, f_window_step, features):
    feature = preprocessing(path_source, f_window_size=f_window_size, f_window_step=f_window_step, features=features)
    feature = feature.T
    feature.to_csv(path_feature_csv, mode='w', index=False)
    writer = pd.ExcelWriter(path_feature_xlsx)
    feature.to_excel(writer, 'Sheet1')
    writer.save()
def preprocessing(path, f_window_size, f_window_step, features):
    feature = dict()
    feature_time = []

    csvfile = open(path)
    gen = csv.reader(csvfile, delimiter=',')
    first_row = gen.__next__()

    index = dict()
    index['seizureStart'] = first_row.index('seizureStart')
    index['seizureStop']  = first_row.index('seizureEnd')
    seizureStart = list()
    seizureStop  = list()
    for row in gen:
        if '' == row[index['seizureStart']]:
            break
        seizureStart.append(float(row[index['seizureStart']]))
        seizureStop.append(float(row[ index['seizureStop']]))


    csvfile       = open(path)
    csvfile_2Hz   = open(path)
    gen           = csv.reader(csvfile, delimiter=',')
    gen_2Hz       = csv.reader(csvfile_2Hz, delimiter=',')
    first_row     = gen.__next__()
    first_row_2Hz = gen_2Hz.__next__()
    index['RR_raw']       = first_row.index('RR_raw')
    index['RR_pos']       = first_row.index('RR_pos')
    index['RR_2Hz']       = first_row_2Hz.index('RR_2Hz')
    index['time_2Hz']     = first_row_2Hz.index('time_2Hz')


    for seizure_counter in range(seizureStart.__len__()+1):
        if seizure_counter == 0:
            interictal_start =  0
        else:
            interictal_start =  seizureStop[seizure_counter - 1 ]
        if seizure_counter == (seizureStart.__len__()):
            interictal_stop = 999999999999999999999999999
        else:
            interictal_stop = seizureStop[seizure_counter]


        RR_raw   = list()
        RR_pos   = list()
        RR_2Hz   = list()
        time_2Hz = list()

        for row in gen:
            if (row[index['RR_pos']] == '' or float(row[index['RR_pos']]) > interictal_stop):
                break
            RR_raw.append(float(row[index['RR_raw']]))
            RR_pos.append(float(row[index['RR_pos']]))

        for row in gen_2Hz:
            if (row[index['time_2Hz']] == '') or (float(row[index['time_2Hz']]) > interictal_stop):
                break
            RR_2Hz.append(float(row[index['RR_2Hz']]))
            time_2Hz.append(float(row[index['time_2Hz']]))

        i = 1
        j = 1
        window = dict()
        window_Hz = dict()
        if( RR_pos.__len__() > 5):
            window_end = max(RR_pos[-1],time_2Hz[-1])
            while True:
                window_begin =  window_end -  f_window_size

                if window_begin < interictal_start:
                    break

                window['RR_raw'] = list()
                window['RR_pos'] = list()
                new_i = i
                while RR_pos[-i] > window_begin:
                    window['RR_raw'].insert(0, RR_raw[-i])
                    window['RR_pos'].insert(0, RR_pos[-i])
                    if RR_pos[-i] > (window_end - f_window_step):
                        new_i = i
                    i += 1
                    if(i == RR_pos.__len__()):
                        break
                i = new_i

                window_Hz['RR_2Hz'] = list()
                window_Hz['time_2Hz'] = list()
                new_j = j
                while time_2Hz[-j] > window_begin:
                    window_Hz['RR_2Hz'].insert(0, RR_2Hz[-j])
                    window_Hz['time_2Hz'].insert(0, time_2Hz[-j])
                    if time_2Hz[-j] > (window_end - f_window_step):
                        new_j = j
                    j+=1
                    if(j == time_2Hz.__len__()):
                        break
                j = new_j

                if window['RR_pos'].__len__() > 5:
                    feature_ext(item=feature, index=feature_time, window=window, window_Hz=window_Hz, features=features)

                window_end = window_end - f_window_step
    wn = list()
    for i in feature['0:f_time']:
        wn.append(float(i))
    ind = np.argsort(wn)
    for i in feature:
        vn = np.array(feature[i])
        vn = vn[ind]
        feature[i] = vn.tolist()
    feature_time = np.array(feature_time)[ind].tolist()

    return pd.DataFrame(feature, index=feature_time)
def feature_ext(item, index, window, window_Hz, features):
    arrays = {i: np.array(window[i]) for i in window}
    arrays_Hz = {i: np.array(window_Hz[i]) for i in window_Hz}

    dataframe_index(index, arrays)
    final_time(item, arrays)
    if (features['length']): feture_len(item, arrays)
    if (features['meanNN']): feature_meanNN(item, arrays)
    if (features['SDNN']): feature_SDNN(item, arrays)
    if (features['RMSSD']): feature_RMSSD(item, arrays)
    if (features['variance']): feature_variance(item, arrays)
    if (features['NN50']): feature_NN50(item, arrays)
    if (features['FDF']): feature_FDF(item, arrays_Hz)
def dataframe_index(index, arrays):
    try:
        index.append(str(int(arrays['RR_pos'][0])) + ':' + str(int(arrays['RR_pos'][-1])))
    except:
        index = [str(int(arrays['RR_pos'][0])) + ':' + str(int(arrays['RR_pos'][-1]))]
def final_time(item, arrays):
    try:
        item['0:f_time'].append(str(int(arrays['RR_pos'][-1])))
    except:
        item['0:f_time'] = [str(int(arrays['RR_pos'][-1]))]
def feture_len(item, arrays):
    try:
        item['1:length'].append(arrays['RR_raw'].size)
    except:
        item['1:length'] = [arrays['RR_raw'].size]
def feature_meanNN(item, arrays):
    try:
        item['2:meanNN'].append(arrays['RR_raw'].mean())
    except:
        item['2:meanNN'] = [arrays['RR_raw'].mean()]
def feature_SDNN(item, arrays):
    try:
        item['3:SDNN'].append(arrays['RR_raw'].std())
    except:
        item['3:SDNN'] = [arrays['RR_raw'].std()]
def feature_RMSSD(item, arrays):
    buf = np.array([(arrays['RR_raw'][i] - arrays['RR_raw'][i - 1]) ** 2 for i in range(1, len(arrays))])
    try:
        item['4:RMSSD'].append(m.sqrt(buf.mean()))
    except:
        item['4:RMSSD'] = [m.sqrt(buf.mean())]
def feature_variance(item, arrays):
    try:
        item['5:variance'].append(arrays['RR_raw'].var())
    except:
        item['5:variance'] = [arrays['RR_raw'].var()]
def feature_NN50(item, arrays):
    buf = [arrays['RR_raw'][i] - arrays['RR_raw'][i - 1] for i in range(1, len(arrays['RR_raw']))]
    buf = np.array([i for i in buf if i > 0.05])
    try:
        item['6:NN50'].append(buf.size)
    except:
        item['6:NN50'] = [buf.size]
def feature_FDF(item, arrays):
    Fs = 2
    N = len(arrays['RR_2Hz'])
    window = np.hanning(N)

    spectrum = np.abs(np.fft.rfft(window * arrays['RR_2Hz']))
    f = np.fft.rfftfreq(N, 1. / Fs)

    powerLF = [spectrum[list(f).index(i)] * spectrum[list(f).index(i)] for i in f if (i > 0.04) & (i < 0.15)]
    powerHF = [spectrum[list(f).index(i)] * spectrum[list(f).index(i)] for i in f if (i > 0.15) & (i < 0.4)]
    try:
        item['LF'].append(sum(powerLF) / len(powerLF))
        item['HF'].append(sum(powerHF) / len(powerHF))
        item['LF/HF'].append(item['LF'][-1] / item['HF'][-1])
    except:
        item['LF'] = [sum(powerLF) / len(powerLF)]
        item['HF'] = [sum(powerHF) / len(powerHF)]
        item['LF/HF'] = [item['LF'][-1] / item['HF'][-1]]
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
def separation( patients,path_features,path_csv,preictal_size,f_window_size,f_window_step,eig,eig_window_size):
    interictal_time    = list()
    interictal_feature = list()
    interictal_w       = list()
    interictal_v       = list()

    preictal_time      = list()
    preictal_feature   = list()
    preictal_w         = list()
    preictal_v         = list()

    for i in range(8):
        interictal_w.append(list())
        interictal_v.append(list())
        preictal_w.append(list())
        preictal_v.append(list())

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

        seizure_counter    = 0
        break_counter      = 0
        interictal_counter = 0
        preictal_counter   = 0
        for i in range(patient_feature.shape[1]):
            break_counter -=1
            if break_counter > 0:
                continue
            if (patient_feature.values[0, i] - patient_feature.values[0, i - 1]) > preictal_size:
                break_counter = int(f_window_size / f_window_step)
                continue

            if ((seizure_start_time - patient_feature.values[0, i]) > preictal_size):
                if((seizure_start_time - patient_feature.values[0, i]) < (4*preictal_size)):
                    preictal_counter = 0
                    interictal_counter += 1
                    interictal_time.append(patient_feature.values[0, i])
                    interictal_feature.append(patient_feature.values[2:, i])
                    if eig:
                        if interictal_counter > (eig_window_size - 1):
                            # window_feature = np.random.uniform(0, 1, (2, 8))
                            window_feature = np.array(interictal_feature[-eig_window_size:])
                            covmat = np.cov(window_feature.T)
                            U, S, V = np.linalg.svd(covmat)
                            wn = S
                            vn = U * S
                            for eig_num in range(8):
                                interictal_w[eig_num].append(wn[eig_num].real)
                                interictal_v[eig_num].append(vn[:, eig_num].real)
            else:
                interictal_counter = 0
                if (seizure_start_time - patient_feature.values[0, i]) > 0:
                    preictal_counter += 1
                    preictal_time.append(patient_feature.values[0, i])
                    preictal_feature.append(patient_feature.values[2:, i])
                    if eig:
                        if preictal_counter > (eig_window_size - 1):

                            # window_feature = np.random.uniform(100, 101, (2, 8))
                            window_feature = np.array(preictal_feature[-eig_window_size:])
                            covmat = np.cov(window_feature.T)
                            U, S, V = np.linalg.svd(covmat)
                            wn = S
                            vn = U*S
                            for eig_num in range(8):
                                preictal_w[eig_num].append(wn[eig_num].real)
                                preictal_v[eig_num].append(vn[:,eig_num].real)

                else:
                    if (patient_feature.values[0, i] - seizure_stop_time) > (f_window_size):
                        seizure_counter += 1
                        if seizure_counter == seizure_start.__len__():
                            seizure_start_time = patient_feature.values[0, -1] + 2 * f_window_size
                            seizure_stop_time = patient_feature.values[0, -1] + 2 * f_window_size
                        else:
                            seizure_start_time = seizure_start[seizure_counter]
                            seizure_stop_time = seizure_stop[seizure_counter]
    if eig:
        return interictal_w, interictal_v,   preictal_w, preictal_v,
    else:
        return interictal_time, interictal_feature, preictal_time ,preictal_feature
def model(interictal_v, preictal_v, train_data, train_part, path_result):
    # fig = plt.figure()
    eig_counter = -1
    ax = [0]
    eig_values = [0]

    for eig_value in eig_values:
        eig_counter += 1
        new_interictal = interictal_v[eig_value]
        new_preictal   = preictal_v[eig_value]
        interictal = list(new_interictal)
        preictal = list(new_preictal)
        arr_inter = [i for i in range(new_interictal.__len__())]
        arr_pre   = [i for i in range(new_preictal.__len__())]
        random.shuffle([i for i in range(new_interictal.__len__())])
        random.shuffle([i for i in range(new_preictal.__len__())])
        for ind in arr_inter:
            interictal.append(new_interictal[ind])
        for ind in arr_pre:
            preictal.append(new_preictal[ind])

        #
        # if train_data is 'interictal':
        #     X_train         = skpp.scale(interictal[:int(train_part * interictal.__len__())])
        #     X_test_train    = skpp.scale(interictal[int(train_part * interictal.__len__()):])
        #     X_test          = skpp.scale(preictal)
        # else:
        #     X_train         = skpp.scale(preictal[:int(train_part * preictal.__len__())])
        #     X_test_train    = skpp.scale(preictal[int(train_part * preictal.__len__()):])
        #     X_test          = skpp.scale(interictal)

        # for i in range(X_train.shape[0]):
        #     X_train[i] = np.random.uniform(0, 1, (1, 8))
        # for i in range(X_test_train.shape[0]):
        #     X_test_train[i] = np.random.uniform(0, 1, (1, 8))
        # for i in range(X_test.shape[0]):
        #     X_test[i] = np.random.uniform(0, 1, (1, 8))

        if train_data is 'interictal':
            X_train         = interictal[:int(train_part * interictal.__len__())]
            X_test_train    = interictal[int(train_part * interictal.__len__()):]
            X_test          = preictal
        else:
            X_train         = preictal[:int(train_part * preictal.__len__())]
            X_test_train    = preictal[int(train_part * preictal.__len__()):]
            X_test          = interictal[int(train_part * interictal.__len__()):]
        nu_plot = list()
        predict_plot_X_test_train = list()
        predict_plot_X_test       = list()
        predict_plot_success      = list()



        # X1 = preictal
        # X2 = interictal
        # y1 = [1 for i in range(preictal.__len__())]
        # y2 = [-1 for i in range(interictal.__len__())]
        # XX = X1[:]
        # yy = y1[:]
        # XX.extend(X2)
        # yy.extend(y2)
        # X = X1[:int(X1.__len__()*0.8)]
        # y = y1[:int(y1.__len__()*0.8)]
        # X.extend(X2[:int(X2.__len__()*0.8)])
        # y.extend(y2[:int(y2.__len__()*0.8)])
        #
        # X1_t =X1[int(X1.__len__()*0.8):]
        # y1_t =y1[int(y1.__len__()*0.8):]
        # X2_t =X2[int(X2.__len__()*0.8):]
        # y2_t =y2[int(y2.__len__()*0.8):]
        #
        # from sklearn.svm import SVC
        # clf = SVC()
        # clf.fit(XX[:int(XX.__len__()*0.2)], yy[:int(yy.__len__()*0.2)])
        # return clf.score(XX[int(XX.__len__()*0.2):], yy[int(yy.__len__()*0.2):])
        # predict_X_test_train            = clf.predict(X1_t)
        # predict_X_test                  = clf.predict(X2_t)
        # prediction_success_X_test_train = 100 * ( X_test_train.__len__() - sum([1 for i in predict_X_test_train if i > 0])) / X_test_train.__len__()
        # prediction_success_X_test       = 100 * ( X_test.__len__() - sum([1 for i in predict_X_test if i < 0])) / X_test.__len__()
        # prediction_success              = (prediction_success_X_test_train + prediction_success_X_test) / 2
        # return prediction_success
        sensitivity = list()
        specificity = list()
        precision   = list()
        F           = list()
        accuracy    = list()
        for nu in range(0,99,1):
            new_interictal = interictal_v[eig_value]
            new_preictal = preictal_v[eig_value]
            interictal = list(new_interictal)
            preictal = list(new_preictal)
            arr_inter = [i for i in range(new_interictal.__len__())]
            arr_pre = [i for i in range(new_preictal.__len__())]
            random.shuffle([i for i in range(new_interictal.__len__())])
            random.shuffle([i for i in range(new_preictal.__len__())])
            for ind in arr_inter:
                interictal.append(new_interictal[ind])
            for ind in arr_pre:
                preictal.append(new_preictal[ind])

            if train_data is 'interictal':
                X_train = interictal[:int(train_part * interictal.__len__())]
                X_test_train = interictal[int(train_part * interictal.__len__()):]
                X_test = preictal
            else:
                X_train = preictal[:int(train_part * preictal.__len__())]
                X_test_train = preictal[int(train_part * preictal.__len__()):]
                X_test = interictal[int(train_part * interictal.__len__()):]

            nu = 0.01 * nu + 0.01
            clf = sksvm.OneClassSVM(nu=nu, kernel='rbf').fit(X_train)

            predict_X_test_train            = clf.predict(X_test_train);
            predict_X_test                  = clf.predict(X_test);

            tp = sum([1 for i in predict_X_test if i < 0])
            tn = sum([1 for i in predict_X_test_train if i > 0])
            fp = len(predict_X_test_train) - tn
            fn = len(predict_X_test) - tp

            sensitivity.append(tp /(tp + fn))  #recall
            specificity.append(tn /(fp + tn))
            precision.append(tp /(tp + fp))
            accuracy.append((tp + tn) /(tp + tn + fn + fp))
            F.append(2 * (precision[-1] * sensitivity[-1]) / (precision[-1] + sensitivity[-1]))
            nu_plot.append(nu*100)
    #         prediction_success_X_test_train = 100 * ( X_test_train.__len__() - sum([1 for i in predict_X_test_train if i < 0])) / X_test_train.__len__()
    #         prediction_success_X_test       = 100 * ( X_test.__len__() - sum([1 for i in predict_X_test if i > 0])) / X_test.__len__()
    #         prediction_success              = (prediction_success_X_test_train + prediction_success_X_test) / 2
    #         print(prediction_success_X_test_train,prediction_success_X_test,prediction_success)
    #
    #         predict_plot_X_test_train.append(prediction_success_X_test_train)
    #         predict_plot_X_test.append(prediction_success_X_test)
    #         predict_plot_success.append(prediction_success)
    #
    #         print('\n')
    #         print('prediction_success_X_test =', prediction_success_X_test,'sensitivity = ', sensitivity[-1])
    #         print('prediction_success_X_test_train =', prediction_success_X_test_train,'specificity = ', specificity[-1])
    #         print('prediction_success =', prediction_success,'precision = ', precision[-1])
    #         print('\n')
    #     red_patch = list()
    #     # ax[eig_counter] = fig.add_subplot(211 + eig_counter)
    #
    #     red_patch.append(mpatches.Patch(label='X_train.__len__ = '          + str(X_train.__len__())))
    #     red_patch.append(mpatches.Patch(label='X_test_train.__len__  = '    + str(X_test_train.__len__())))
    #     red_patch.append(mpatches.Patch(label='X_test.__len__  = '          + str(X_test.__len__())))
    #
    #     plt.plot(nu_plot, predict_plot_X_test_train, 'b')
    #     if train_data is 'interictal':
    #         red_patch.append(mpatches.Patch(color='blue', label='Interictal   accuracy'))
    #     else:
    #         red_patch.append(mpatches.Patch(color='blue', label='Preictal   accuracy'))
    #
    #     plt.plot(nu_plot, predict_plot_X_test, 'g')
    #     if train_data is 'interictal':
    #         red_patch.append(mpatches.Patch(color='green', label='Preictal   accuracy'))
    #     else:
    #         red_patch.append(mpatches.Patch(color='green', label='Interictal   accuracy'))
    #
    #     plt.plot(nu_plot, predict_plot_success, 'r')
    #     red_patch.append(mpatches.Patch(color='red', label='Prediction accuracy'))
    #
    #     # if eig_value == 7:
    #         # plt.title('eigvalue is minimal')
    #     # if eig_value == 0:
    #         # plt.title('eigvalue is maximal')
    #
    #     plt.legend(handles=red_patch, loc=4)
    #     plt.ylabel('Prediction accuracy, %')
    #     plt.xlabel('Fraction of training errors ,%')
    #     plt.grid(True)
    #     plt.axis([0, 100, -1, 101])
    #
    # # fig.set_size_inches(12, 20)
    # plt.savefig(path_result)
    # plt.close(fig)


    fig = plt.figure()

    red_patch = list()
    plt.plot(nu_plot, sensitivity, 'b')
    red_patch.append(mpatches.Patch(color='blue', label='sensitivity'))
    plt.plot(nu_plot, specificity, 'g')
    red_patch.append(mpatches.Patch(color='green', label='specificity'))
    plt.plot(nu_plot, precision, 'r')
    red_patch.append(mpatches.Patch(color='red', label='precision'))
    plt.plot(nu_plot, F, 'y')
    red_patch.append(mpatches.Patch(color='yellow', label='F'))
    plt.plot(nu_plot, accuracy, 'black')
    red_patch.append(mpatches.Patch(color='black', label='accuracy'))
    plt.legend(handles=red_patch, loc=4)
    plt.ylabel('-, %')
    plt.xlabel('Fraction of training errors ,%')
    plt.grid(True)
    # plt.axis([0, 100, -1, 101])

    plt.savefig(path_result.split('Results_article/')[0] + 'Results_article/' + 'new' + path_result.split('Results_article/')[1])
    plt.close(fig)
    # return 1
    Fa = np.array(F[0:30])
    Fm = np.mean(Fa)
    print(np.mean(np.array(sensitivity[0:30])),np.mean(np.array(specificity[0:30])),np.mean(np.array(precision[0:30])),np.mean(np.array(F[0:30])),np.mean(np.array(accuracy[0:30])) )
    return Fm


feature_sizes = [20*i for i in range(3,33,1)]

# feature_sizes = [280]
# model settings
window = dict()
window ['eig_size']      = []
window ['feature_size']  = []
window ['feature_step']  = []
window ['preictal_size'] = []
window ['train_datas']   = []
window ['train_parts']   = []

# HRV types
ecg_types = dict()
ecg_types['focal']       = 'focal_seizures/'
ecg_types['generalized'] = 'generalized_seizures/'

# path dict
path = dict()
path['signal'] = '/media/smirnov/Новый том/epilepsy/signals/'
path['feature'] = '/media/smirnov/Новый том/epilepsy/exp_3/Features/'
path['result'] = '/media/smirnov/Новый том/epilepsy/exp_3/Results_article/'

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
                            'bernatskaya1-5','bernatskaya-15',
                            'butkevych-15',
                            'gudz-5',
                            'kaganiuk-15',
                            'klavdiev-15',
                            'kozubal-13',
                            'lekhkodukh-15',
                            'mazanichka-15',
                            'nesterchuk1-14','nesterchuk2-15','nesterchuk3-15','nesterchuk4-15',
                            'podvizhenko-15',
                            'polulyah-14-15',
                            'reshetnik-15',
                            'serdich1-15','serdich-14',
                            'stupak-14',
                            'tarasov-18',
                            'tumareva-14-15',
                            'volkogon-15'
                            ]
patients ['generalized'] =  [
                            'drozdov-14',
                            'dukova-15',
                            'feschenko-15',
                            'karpenko-15',
                            'kharchenko-15-16',
                            'kostuk-15',
                            'levchenko1_26_01_14-15', 'levchenko2_21_06_14-14-15-16',
                            'marchenko1-15', 'marchenko-15',
                            'petrian-15',
                            'semashko1-14', 'semashko-14', 'semashko2-14',
                            'shamina-15',
                            'surdu-15',
                            'yakimets-14'
                            ]


# ############################################################################
# #                               Test #8 30 1                               #
# ############################################################################
for feature_size in feature_sizes:
    window ['eig_size'].append(2)
    window ['feature_size'].append(feature_size)
    window ['feature_step'].append(feature_size)
    window ['preictal_size'].append(2*feature_size)
    window ['train_datas'].append('interictal')
    window ['train_parts'].append(0.8)

############################################################################
#                   Feature selections                                     #
############################################################################
print('Feature selections')
path_feature_csv = dict()
path_feature_xlsx = dict()
for ecg_type in ecg_types:
    for patient in patients[ecg_type]:
        print(patient)
        path_source = path['signal'] + ecg_types[ecg_type] + patient + '.csv'
        for i in range(window['feature_size'].__len__()):
            if i>-1:
                feature_size = window['feature_size'][i]
                feature_step = window['feature_step'][i]

                path_feature_csv  = path['feature']+ ecg_types[ecg_type] + 'csv/'   + str(feature_size) + '_' + str(feature_step) + '_' + patient + '.csv'
                path_feature_xlsx = path['feature']+ ecg_types[ecg_type] + 'excel/' + str(feature_size) + '_' + str(feature_step) + '_' + patient + '.xlsx'

                # print(str(feature_size) + '   ' + str(feature_step))
                # new_features(path_source,path_feature_csv,path_feature_xlsx, feature_size,feature_step,features)
############################################################################
#                   Model                                                  #
############################################################################
window = dict()
window ['eig_size']      = []
window ['feature_size']  = []
window ['feature_step']  = []
window ['preictal_size'] = []
window ['train_datas']   = []
window ['train_parts']   = []

for feature_size in feature_sizes:
    for preictal_size in [2*feature_size]:
        window ['eig_size'].append(2)
        window ['feature_size'].append(feature_size)
        window ['feature_step'].append(feature_size)
        window ['preictal_size'].append(preictal_size)
        window ['train_datas'].append('interictal')
        window ['train_parts'].append(0.8)

print('Model')
path_features  = dict()
path_sources   = dict()
for i in range(window['feature_size'].__len__()):
    eig_size       = window['eig_size'][i]
    feature_size   = window['feature_size'][i]
    feature_step   = window['feature_step'][i]
    preictal_size  = window['preictal_size'][i]
    train_data     = window['train_datas'][i]
    train_part     = window['train_parts'][i]

    for train_data in ['interictal']:
        path_features  = path['feature']+ ecg_types['focal'] + 'csv/'   + str(feature_size) + '_' + str(feature_step) + '_'
        path_sources   = path['signal'] + ecg_types['focal']

        f_interictal_w, f_interictal_v,   f_preictal_w, f_preictal_v  = separation(   patients['focal'],
                                                                                                                                        eig             = True,
                                                                                                                                        path_features   = path_features,
                                                                                                                                        path_csv        = path_sources,
                                                                                                                                        preictal_size   = preictal_size,
                                                                                                                                        f_window_size   = feature_size,
                                                                                                                                        f_window_step   = feature_step,
                                                                                                                                        eig_window_size = eig_size
                                                                                                                                    )

        path_features  = path['feature']+ ecg_types['generalized'] + 'csv/'   + str(feature_size) + '_' + str(feature_step) + '_'
        path_sources   = path['signal'] + ecg_types['generalized']
        interictal_w, interictal_v,   preictal_w, preictal_v  = separation(   patients['generalized'],
                                                                                                                                        eig             = True,
                                                                                                                                        path_features   = path_features,
                                                                                                                                        path_csv        = path_sources,
                                                                                                                                        preictal_size   = preictal_size,
                                                                                                                                        f_window_size   = feature_size,
                                                                                                                                        f_window_step   = feature_step,
                                                                                                                                        eig_window_size = eig_size
                                                                                                                                    )

        for j in range(8):
            for i in range(f_preictal_v[j].__len__()):
                preictal_v[j].append(f_preictal_v[j][i])
        for j in range(8):
            for i in range(f_interictal_v[j].__len__()):
                interictal_v[j].append(f_interictal_v[j][i])

        path_result = path['result']
        try:
            os.makedirs(path_result, 493)
            path_result =path_result + '/' +   train_data + '_' + str(feature_size)  + '_' + str(eig_size) + '_' + str(feature_step) + '_' + str(preictal_size) + '.png'
        except:
            path_result = path_result + '/' +   train_data + '_' + str(feature_size)  + '_' + str(eig_size) + '_' + str(feature_step) + '_' + str(preictal_size) + '.png'

        try:
            res.append(model(list(interictal_v),list(preictal_v), train_data, train_part, str(path_result)))
        except:
            res = [model(list(interictal_v),list(preictal_v), train_data, train_part, str(path_result))]

        print('TEST: '  + train_data + '_' + str(feature_size)  + '_' + str(eig_size) + '_' + str(feature_step) + '_' + str(preictal_size) +  '    FINISHED')
print(res)
plt.plot(feature_sizes, res, 'b')
plt.ylabel('F')
plt.xlabel('Window size, sec')
# plt.axis([0, 750, 50, 100])
plt.grid(True)
plt.savefig(path['result']+'/final1.png')

# hist_data = [ 1+res.index(i) for i in res for j in range(int(i)) ]
# plt.hist(hist_data,[0.9,1.9,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26])
# plt.title("Gaussian Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()