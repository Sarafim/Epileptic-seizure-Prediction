import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.svm as sksvm

import load as ld
import preprocessing as pp

# sources and path
path_mat      = 'D:/Diploma/MAT/'
path_csv      = 'D:/Diploma/Signals/'
path_features = 'D:/Diploma/Features/f_'
path_results  = 'D:/Diploma/Results/'

# Patients
patients=['boyko-18',
          'bernatskaya1-5','bernatskaya-15',
          'butkevych-15']

# Settings
LOAD_MAT_FILE  = False
SELECT_FEATURE = False
f_window_size = 300
f_window_step  = 60
features   = dict()
features['length']  = True
features['meanNN']  = True
features['SDNN']    = True
features['RMSSD']   = True
features['variance']= True
features['NN50']    = True
features['FDF']     = True

eig_window_size = 4
eig_window_step = 1
eig_num = 0

############################################################################
#  .mat into .csv
if LOAD_MAT_FILE:
    for patient in patients:
        item = ld.load_mat(path_mat + patient + '.mat')
        item.to_csv(path_csv + patient + '.csv', mode = 'w',index=False)
############################################################################
# feature selections
if SELECT_FEATURE:
    for patient in patients:
        path_source       = path_csv      + patient + '.csv'
        path_feature_csv  = path_features + patient + '.csv'
        path_feature_xlsx = path_features + patient + '.xlsx'
        pp.new_features(path_source,path_feature_csv,path_feature_xlsx, f_window_size, f_window_step,features)
############################################################################
# creating dataset
interictal_w,interictal_v,preictal_w,preictal_v = pp.eig_separation(patients,
                                                                    eig = True,
                                                                    path_features = 'D:/Diploma/Features/f_',
                                                                    path_csv = 'D:/Diploma/Signals/',
                                                                    f_window_size = 300,
                                                                    f_window_step = 60,
                                                                    eig_window_size = 4,
                                                                    eig_num = 0)
interictal_time,interictal_feature,preictal_time,preictal_feature = pp.eig_separation(patients,
                                                                                      eig = False,
                                                                                      path_features = 'D:/Diploma/Features/f_',
                                                                                      path_csv = 'D:/Diploma/Signals/',
                                                                                      f_window_size = 300,
                                                                                      f_window_step = 60,
                                                                                      eig_window_size = 4,
                                                                                      eig_num = 0)
############################################################################
# Model
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
for ker in kernel:
    clf = sksvm.OneClassSVM(nu = 0.1,kernel=ker).fit(interictal_v)
    predict = clf.predict(preictal_v);
    # Success calculation
    total_success = sum([1 for i in predict if i<0])/predict.__len__()
    prediction_success = sum([(predict[i]<0) or (predict[i-1]<0) for i in range(predict.__len__()) if int(i%2)])/(predict.__len__()/2)
    print(ker)
    print('total_success      = ',total_success )
    print('prediction_success = ',prediction_success )
print(6)
# # /////////////////////////////////////////////////////
# TIME = ld.load_column(path_s, data_type='time')
# DATA = ld.load_column(path_s, data_type='data')
# RR_raw = ld.load_column(path_s, data_type='RR_raw')
# RR_pos = ld.load_column(path_s, data_type='RR_pos')

# ax = [0,1,2]
# fig = plt.figure()
# ax[0] = fig.add_subplot(111 + 0)
# ax[0].plot(TIME['time'],DATA['data'])
# ax[0].plot([x/100  for x in sS['seizureStart']], np.zeros(len(sS['seizureStart'])), marker='o',linestyle=' ', color='r')
# ax[0].title.set_text('ECG')
# ax[0].grid(True)
# fig.set_size_inches(25, 20)
# fig.savefig('D:/Diploma/ECG_' + str(test) + '_' + str(f_window_size) + '_' + str(f_window_step) + '_' + str(eig_step) + '_' + str(eig_num) + '.png')  # save the figure to file
# plt.close(fig)
# ///////////////////////////////////////////////////////////////

#
# clf = sksvm.OneClassSVM(nu = 0.1,kernel='rbf').fit(v)
# predict = clf.predict(v);
# chance = clf.decision_function(v).ravel()
#
# fig = plt.figure()
# time = [(f_window_size + f_window_step*(i+4))/100 for i in range(ind+1)]
# tick = range(0,150,2)
# red_patch = list()
# for i in range(sS['seizureStart'].__len__()):
#     red_patch.append(mpatches.Patch(label='SeizureStart : SeizureEnd  ' + str(sS['seizureStart'][i]/100.0) +' : ' +   str(sE['seizureEnd'][i]/100.0)))
# # ax = [0,1,2,3,4,5,6,7]
# # for num in range(7):
# #     ax[num] = fig.add_subplot(711 + num)
# #     plt.legend(handles=red_patch,loc=2)
# #     ax[num].plot(time,w[num])
# #     ax[num].plot([x/100  for x in sS['seizureStart']], np.zeros(len(sS['seizureStart'])), marker='o',linestyle=' ', color='r')
# #     ax[num].title.set_text('Change of the ' +str(num) + ' Eigenvalue')
# #     ax[num].xaxis.set_ticks(list(tick))
# #     ax[num].grid(True)
#
# # fig.set_size_inches(25, 80)
#
# ax = [0,1,2,3,4]
# ax[0] = fig.add_subplot(511 + 0)
# plt.legend(handles=red_patch,loc=2)
# ax[0].plot(time,w[eig_num])
# ax[0].plot([x/100  for x in sS['seizureStart']], np.zeros(len(sS['seizureStart'])), marker='o',linestyle=' ', color='r')
# ax[0].title.set_text('Change of the ' +str(0) + ' Eigenvalue')
# ax[0].xaxis.set_ticks(list(tick))
# ax[0].grid(True)
#
# ax[1] = fig.add_subplot(511 + 1)
# ax[1].plot(time,predict)
# ax[1].plot([x/100  for x in sS['seizureStart']], np.zeros(len(sS['seizureStart'])), marker='o',linestyle=' ', color='r')
# ax[1].title.set_text('Decision_function, ' +str(0) + ' Eigenvalue')
# ax[1].xaxis.set_ticks(list(tick))
# ax[1].grid(True)
#
# ax[2] = fig.add_subplot(511 + 2)
# ax[2].plot(time,chance)
# ax[2].plot([x/100  for x in sS['seizureStart']], np.zeros(len(sS['seizureStart'])), marker='o',linestyle=' ', color='r')
# ax[2].title.set_text('Change of the ' +str(0) + ' Eigenvalue')
# ax[2].xaxis.set_ticks(list(tick))
# ax[2].axis([0, 148,-0.05, 0.05])
# ax[2].grid(True)
#
# # fig.set_size_inches(25, 20)
# # fig.savefig('D:/Diploma/' + str(test) + '_' + str(f_window_size) + '_' + str(f_window_step) + '_' + str(eig_step) + '_' + str(eig_num) + '.png')  # save the figure to file
# # plt.close(fig)
#
# ax[3] = fig.add_subplot(511 + 3)
# ax[3].plot(TIME['time'],DATA['data'])
# ax[3].plot(sS['seizureStart'], np.zeros(len(sS['seizureStart'])), marker='o',linestyle=' ', color='r')
# ax[3].grid(True)
#
# ax[4] = fig.add_subplot(511 + 4)
# ax[4].plot(RR_pos['RR_pos'],RR_raw['RR_raw'])
# ax[4].plot(sS['seizureStart'], np.zeros(len(sS['seizureStart'])), marker='o',linestyle=' ', color='r')
# ax[4].grid(True)
#
# fig.set_size_inches(60, 30)
# fig.savefig('D:/Diploma/ECG_' + str(test) + '_' + str(f_window_size) + '_' + str(f_window_step) + '_' + str(eig_step) + '_' + str(eig_num) + '.png')  # save the figure to file
# plt.close(fig)
#
#
#
# w = list(w)
# w=pd.DataFrame({'1':w[0],'2':w[1],'3':w[2],'4':w[3],'5':w[4],'6':w[5],'7':w[6],'8':w[7]})
# writer = pd.ExcelWriter('D:/Diploma/Features/w_stupak-14.xlsx')
# w.to_excel(writer, 'Sheet1')
# writer.save()
# print(ind)

# строки -  по фичерам, 8 строкs
