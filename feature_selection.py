###################################################################################################
#   feature selection
###################################################################################################
# patients      	= ['patient_name']
# path_csv      	= 'C:/path_to_cvsfile/'
# path_features 	= 'C:/path_to_new_file_with_feature/'
# path_source       = path_csv + patient + '.csv'
# path_feature_csv  = path_features + patient + '.csv'
# path_feature_xlsx = path_features + patient + '.xlsx'
#
# f_window_size  = 300
# f_window_step  = 60
#
# features            = dict()
# features['length']  = True
# features['meanNN']  = True
# features['SDNN']    = True
# features['RMSSD']   = True
# features['variance']= True
# features['NN50']    = True
# features['FDF']     = True
#
# new_features(path_source,path_feature_csv,path_feature_xlsx, f_window_size, f_window_step,features)
###################################################################################################
import numpy as np
import pandas as pd
import math as m
import csv


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
            window_begin = max(RR_pos[-1],time_2Hz[-1])
            while True:
                window_end   =  window_begin
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


def load_data(path, data_type):
    csvfile = open(path)
    gen = csv.reader(csvfile, delimiter=',')
    row_first = gen.__next__()
    ind = [row_first.index(i) for i in data_type]
    stop_flag = 0

    def prog_load(item, step_add, step_del, mode='all'):
        '''
        return  1 reach end of the file at this step
        return -1 end was reached before
        return  0 all is ok
        '''
        nonlocal stop_flag

        count_step = 1
        if stop_flag == 1 or stop_flag == -1:
            stop_flag = -1
            return stop_flag

        for row in gen:
            for i in ind:
                if '' == row[i]:
                    stop_flag = 1
                    return stop_flag

                try:
                    item[row_first[i]].append(float(row[i]))
                except:
                    item[row_first[i]] = [float(row[i])]
            if mode != 'all':
                if step_add == count_step:
                    if mode == 'step':
                        for i in ind:
                            del item[row_first[i]][:step_del]
                    return stop_flag
                else:
                    count_step += 1
        else:
            for i in ind:
                del item[row_first[i]][:step_del - 1]
            stop_flag = 1
            return stop_flag

    return prog_load


def step_del(path, data_type, f_window_step=60):
    csvfile = open(path)
    gen = csv.reader(csvfile, delimiter=',')
    row_first = gen.__next__()
    ind = row_first.index(data_type)
    yield 0
    step = 1
    count_time = f_window_step
    for row in gen:
        if row[ind] == '':
            yield step
        if float(row[ind]) > count_time:
            yield step
            count_time += f_window_step
            step = 1
        else:
            step += 1
    else:
        return step


def step_add(path, data_type, f_window_size=300, f_window_step=60):
    csvfile = open(path)
    gen = csv.reader(csvfile, delimiter=',')
    row_first = gen.__next__()
    ind = row_first.index(data_type)
    step = 1
    for row in gen:
        if row[ind] == '':
            yield step
        if float(row[ind]) > f_window_size:
            yield step
            f_window_size += f_window_step
            step = 1
        else:
            step += 1
    else:
        return step












