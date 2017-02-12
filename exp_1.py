from feature_selection      import new_features
from feature_preproccesing 	import interictal, preictal

import sklearn.preprocessing as skpp
import sklearn.svm           as sksvm
import matplotlib.pyplot     as plt
import matplotlib.patches    as mpatches

# model 
def model(X_train,X_test,train_data,path_results,file_name):
    # if scale:
    fig = plt.figure()  
    ax  = [0,1,3,4]

    for j in range(2):
        print('scale = {0:1d}'.format(j))


        if j:
            X_train = skpp.scale(X_train)
            X_test  = skpp.scale(X_test)
        # pure accuracy
        nu_plot      = list()
        predict_plot = list()
        for nu in range(50):
            nu = 0.01*nu+0.01
            clf = sksvm.OneClassSVM(nu = nu,kernel='rbf').fit(X_train)
            predict = clf.predict(X_test);
            prediction_success = 100 * (X_test.__len__() - sum([ 1 for i in predict if i > 0]))/X_test.__len__()
    
            red_patch = list()
            red_patch.append(mpatches.Patch(label='X_train.size = ' + str(X_train.__len__())))
            red_patch.append(mpatches.Patch(label='X_test.size  = ' + str(X_test.__len__())))
            nu_plot.append(nu)
            predict_plot.append(prediction_success)

        ax[2*j] = fig.add_subplot(411 + 2*j)
        if j:
            ax[2*j].title.set_text('Test only interictal intervals.Scaling is ON')
        else:
            ax[2*j].title.set_text('Test only interictal intervals.Scaling is OFF')
        ax[2*j].plot(nu_plot,predict_plot)
        plt.legend(handles=red_patch,loc=4)
        ax[2*j].set_ylabel('Prediction accuracy, %')
        ax[2*j].set_xlabel('Fraction of training errors,%')
        ax[2*j].grid(True)

        # better accuracy
        nu_plot             = list()
        predict_plot_pre    = list()
        predict_plot_inter  = list()
        predict_plot        = list()
        X_train_pre   = X_train [:int(0.8*X_train.__len__())]
        X_test_inter  = X_test  [:]
        X_test_pre    = X_train [int(0.8*X_train.__len__()):]
        for nu in range(50):
            nu = 0.01*nu+0.01
            clf = sksvm.OneClassSVM(nu = nu,kernel='rbf').fit(X_train_pre)
            predict_inter = clf.predict(X_test_inter);
            predict_pre   = clf.predict(X_test_pre);
            prediction_success_inter = 100 * (X_test_inter.__len__() - sum([ 1 for i in predict_inter if i > 0]))/X_test_inter.__len__()
            prediction_success_pre   = 100 * (X_test_pre.__len__() - sum([ 1 for i in predict_pre if i < 0]))/X_test_pre.__len__()
            prediction_success       = (prediction_success_inter + prediction_success_pre)/2
            red_patch = list()
            ax[2 * j + 1] = fig.add_subplot(411 + 2 * j + 1)
            if train_data is 'pre':
                red_patch.append(mpatches.Patch(label='X_train.size = ' + str(X_train_pre.__len__())))
                red_patch.append(mpatches.Patch(label='X_test_inter.size  = ' + str(X_test_inter.__len__())))
                red_patch.append(mpatches.Patch(label='X_test_pre.size  = ' + str(X_test_pre.__len__())))
                ax[2 * j + 1].plot(nu_plot, predict_plot_pre, 'b')
                ax[2 * j + 1].plot(nu_plot, predict_plot_inter, 'g')
                ax[2 * j + 1].plot(nu_plot, predict_plot, 'r')
            if train_data is 'inter':
                red_patch.append(mpatches.Patch(label='X_train.size = ' + str(X_train_pre.__len__())))
                red_patch.append(mpatches.Patch(label='X_test_pre.size  = ' + str(X_test_inter.__len__())))
                red_patch.append(mpatches.Patch(label='X_test_inter.size  = ' + str(X_test_pre.__len__())))
                ax[2 * j + 1].plot(nu_plot, predict_plot_pre, 'g')
                ax[2 * j + 1].plot(nu_plot, predict_plot_inter, 'b')
                ax[2 * j + 1].plot(nu_plot, predict_plot, 'r')
            red_patch.append(mpatches.Patch(color = 'blue',  label  = 'Preictal accuracy'))
            red_patch.append(mpatches.Patch(color = 'green', label  = 'Interictal   accuracy'))
            red_patch.append(mpatches.Patch(color = 'red',   label  = 'Prediction accuracy'))

            nu_plot.append(nu)
            predict_plot_pre.append(prediction_success_pre)
            predict_plot_inter.append(prediction_success_inter)
            predict_plot.append(prediction_success)

        if j:
            ax[2*j+1].title.set_text('Test only inter- and preictal intervals.Scaling is ON')
        else:
            ax[2*j+1].title.set_text('Test only inter- and preictal intervals.Scaling is OFF')

        plt.legend(handles=red_patch,loc=4)
        ax[2*j+1].set_ylabel('Prediction accuracy, %')
        ax[2*j+1].set_xlabel('Fraction of training errors,%')
        ax[2*j+1].grid(True)


    fig.set_size_inches(20, 30)
    fig.savefig(path_results + train_data.upper() + '_' + file_name + '.png')
    plt.close(fig)

# experiments
EXP_FOCAL_FEATURES      = True
FOCAL_FEATURES          = True

EXP_FOCAL_EIG           = True
FOCAL_EIG               = True

EXP_GENERALIZED_FEATURES= True
GENERALIZED_FEATURES    = True

EXP_GENERALIZED_EIG     = True
GENERALIZED_EIG         = True

EXP_COMBINED_FEATURES   = True
EXP_COMBINED_EIG        = True

# settings
WINDOW = '720'
if  WINDOW is '360':
# sources and path
    path_focal_csv            = 'D:/Diploma/Signals/Focal_Seizures_processed/'
    path_gener_csv            = 'D:/Diploma/Signals/Generalized_Seizures_processed/'

    path_focal_inter_features = 'D:/Diploma/Features/360/focal_inter_features/'
    path_focal_pre_features   = 'D:/Diploma/Features/360/focal_pre_features/'

    path_focal_inter_eig      = 'D:/Diploma/Features/360/focal_inter_eig/'
    path_focal_pre_eig        = 'D:/Diploma/Features/360/focal_pre_eig/'

    path_gener_inter_features = 'D:/Diploma/Features/360/gener_inter_features/'
    path_gener_pre_features   = 'D:/Diploma/Features/360/gener_pre_features/'

    path_gener_inter_eig      = 'D:/Diploma/Features/360/gener_inter_eig/'
    path_gener_pre_eig        = 'D:/Diploma/Features/360/gener_pre_eig/'

    path_results              = 'D:/Diploma/Results/360/'

    # windows settings
    feature_inter_window_size     = 360
    feature_inter_window_step     = 60
    feature_inter_preictal_size   = 360
    feature_inter_eig_window_size = 2
    feature_inter_eig_num         = 0

    feature_pre_window_size       = 360
    feature_pre_window_step       = 10
    feature_pre_interval          = 6
    feature_pre_eig_window_size   = 2
    feature_pre_eig_num           = 0

    eig_inter_window_size         = 300
    eig_inter_window_step         = 60
    eig_inter_preictal_size       = 360
    eig_inter_eig_window_size     = 2
    eig_inter_eig_num             = 0

    eig_pre_window_size           = 300
    eig_pre_window_step           = 10
    eig_pre_interval              = 6
    eig_pre_eig_window_size       = 2
    eig_pre_eig_num               = 0

if WINDOW is '120':
    path_focal_csv            = 'D:/Diploma/Signals/Focal_Seizures_processed/'
    path_gener_csv            = 'D:/Diploma/Signals/Generalized_Seizures_processed/'

    path_focal_inter_features = 'D:/Diploma/Features/120/focal_inter_features/'
    path_focal_pre_features   = 'D:/Diploma/Features/120/focal_pre_features/'

    path_focal_inter_eig      = 'D:/Diploma/Features/120/focal_inter_eig/'
    path_focal_pre_eig        = 'D:/Diploma/Features/120/focal_pre_eig/'

    path_gener_inter_features = 'D:/Diploma/Features/120/gener_inter_features/'
    path_gener_pre_features   = 'D:/Diploma/Features/120/gener_pre_features/'

    path_gener_inter_eig      = 'D:/Diploma/Features/120/gener_inter_eig/'
    path_gener_pre_eig        = 'D:/Diploma/Features/120/gener_pre_eig/'

    path_results              = 'D:/Diploma/Results/120/'

    # windows settings
    feature_inter_window_size     = 120
    feature_inter_window_step     = 60
    feature_inter_preictal_size   = 120
    feature_inter_eig_window_size = 2
    feature_inter_eig_num         = 0

    feature_pre_window_size       = 120
    feature_pre_window_step       = 10
    feature_pre_interval          = 6
    feature_pre_eig_window_size   = 2
    feature_pre_eig_num           = 0

    eig_inter_window_size         = 100
    eig_inter_window_step         = 50
    eig_inter_preictal_size       = 120
    eig_inter_eig_window_size     = 2
    eig_inter_eig_num             = 0

    eig_pre_window_size           = 100
    eig_pre_window_step           = 10
    eig_pre_interval              = 6
    eig_pre_eig_window_size       = 2
    eig_pre_eig_num               = 0

if WINDOW is '720':
    path_focal_csv            = 'D:/Diploma/Signals/Focal_Seizures_processed/'
    path_gener_csv            = 'D:/Diploma/Signals/Generalized_Seizures_processed/'

    path_focal_inter_features = 'D:/Diploma/Features/720/focal_inter_features/'
    path_focal_pre_features   = 'D:/Diploma/Features/720/focal_pre_features/'

    path_focal_inter_eig      = 'D:/Diploma/Features/720/focal_inter_eig/'
    path_focal_pre_eig        = 'D:/Diploma/Features/720/focal_pre_eig/'

    path_gener_inter_features = 'D:/Diploma/Features/720/gener_inter_features/'
    path_gener_pre_features   = 'D:/Diploma/Features/720/gener_pre_features/'

    path_gener_inter_eig      = 'D:/Diploma/Features/720/gener_inter_eig/'
    path_gener_pre_eig        = 'D:/Diploma/Features/720/gener_pre_eig/'

    path_results              = 'D:/Diploma/Results/720/'

    # windows settings
    feature_inter_window_size     = 720
    feature_inter_window_step     = 60
    feature_inter_preictal_size   = 720
    feature_inter_eig_window_size = 2
    feature_inter_eig_num         = 0

    feature_pre_window_size       = 720
    feature_pre_window_step       = 20
    feature_pre_interval          = 3
    feature_pre_eig_window_size   = 2
    feature_pre_eig_num           = 0

    eig_inter_window_size         = 660
    eig_inter_window_step         = 60
    eig_inter_preictal_size       = 720
    eig_inter_eig_window_size     = 2
    eig_inter_eig_num             = 0

    eig_pre_window_size           = 660
    eig_pre_window_step           = 20
    eig_pre_interval              = 3
    eig_pre_eig_window_size       = 2
    eig_pre_eig_num               = 0

# features list
features            = dict()
features['length']  = True
features['meanNN']  = True
features['SDNN']    = True
features['RMSSD']   = True
features['variance']= True
features['NN50']    = True
features['FDF']     = True

# patients
patients_focal = [
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

patients_gener = [
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
############################################################################
#                   EXPERIMENT WITH FOCAL FEATURES                         #
############################################################################
if EXP_FOCAL_FEATURES:
    file_name = 'EXPERIMENT WITH FOCAL FEATURES'
    print('EXPERIMENT WITH FOCAL FEATURES')
    # focal interictal feature
    path_features   = path_focal_inter_features
    f_window_size   = feature_inter_window_size
    f_window_step   = feature_inter_window_step
    preictal_size   = feature_inter_preictal_size
    eig_window_size = feature_inter_eig_window_size
    eig_num         = feature_inter_eig_num
    
    if FOCAL_FEATURES:
        for patient in patients_focal:
            print(patient)
            path_source       = path_focal_csv+ patient + '.csv'
            path_feature_csv  = path_features + patient + '.csv'
            path_feature_xlsx = path_features + patient + '.xlsx'
            new_features(path_source,path_feature_csv,path_feature_xlsx, f_window_size, f_window_step,features)

    focal_interictal_time,focal_interictal_feature 	= interictal(  	patients_focal,
                                                                    eig             = False,
                                                                    path_features 	= path_features,
                                                                    path_csv 		= path_focal_csv,
                                                                    preictal_size   = preictal_size,
                                                                    f_window_size 	= f_window_size,
                                                                    f_window_step 	= f_window_step,
                                                                    eig_window_size = eig_window_size,
                                                                    eig_num 		= eig_num
                                                                )
    # focal preictal feature 
    path_features   = path_focal_pre_features
    f_window_size   = feature_pre_window_size
    f_window_step   = feature_pre_window_step
    interval        = feature_pre_interval
    eig_window_size = feature_pre_eig_window_size
    eig_num         = feature_pre_eig_num
    if FOCAL_FEATURES:
        for patient in patients_focal:
            print(patient)
            path_source       = path_focal_csv+ patient + '.csv'
            path_feature_csv  = path_features + patient + '.csv'
            path_feature_xlsx = path_features + patient + '.xlsx'
            new_features(path_source,path_feature_csv,path_feature_xlsx, f_window_size, f_window_step,features)

    focal_preictal_time,focal_preictal_feature  = preictal(     patients_focal,
                                                                eig             = False,
                                                                path_features   = path_features,
                                                                path_csv        = path_focal_csv,
                                                                f_window_size   = f_window_size,
                                                                f_window_step   = f_window_step,
                                                                eig_num         = eig_num,
                                                                interval        = interval,
                                                                eig_window_size = eig_window_size
                                                            )
    # focal features model
    model(X_train = list(focal_preictal_feature), X_test = list(focal_interictal_feature), train_data='pre',path_results = path_results,file_name = file_name)
    model(X_train = list(focal_interictal_feature), X_test = list(focal_preictal_feature), train_data='inter',path_results = path_results,file_name = file_name)

############################################################################
#                   EXPERIMENT WITH FOCAL EIG VECTOR                       #
############################################################################
if EXP_FOCAL_EIG:
    file_name = 'EXPERIMENT WITH FOCAL EIG VECTOR'
    print('EXPERIMENT WITH FOCAL EIG VECTOR')
    # focal interictal eig vector
    path_features   = path_focal_inter_eig
    f_window_size   = eig_inter_window_size
    f_window_step   = eig_inter_window_step
    preictal_size   = eig_inter_preictal_size
    eig_window_size = eig_inter_eig_window_size
    eig_num         = eig_inter_eig_num
    
    if FOCAL_EIG:
        for patient in patients_focal:
            print(patient)
            path_source       = path_focal_csv+ patient + '.csv'
            path_feature_csv  = path_features + patient + '.csv'
            path_feature_xlsx = path_features + patient + '.xlsx'
            new_features(path_source,path_feature_csv,path_feature_xlsx, f_window_size, f_window_step,features)

    focal_interictal_w,focal_interictal_v 	        = interictal(  	patients_focal,
                                                                    eig             = True,
                                                                    path_features 	= path_features,
                                                                    path_csv 		= path_focal_csv,
                                                                    preictal_size   = preictal_size,
                                                                    f_window_size 	= f_window_size,
                                                                    f_window_step 	= f_window_step,
                                                                    eig_window_size = eig_window_size,
                                                                    eig_num 		= eig_num
                                                                )
    # focal preictal eig vector
    path_features   = path_focal_pre_eig
    f_window_size   = eig_pre_window_size
    f_window_step   = eig_pre_window_step
    interval        = eig_pre_interval
    eig_window_size = eig_pre_eig_window_size
    eig_num         = eig_pre_eig_num
    if FOCAL_EIG:
        for patient in patients_focal:
            print(patient)
            path_source       = path_focal_csv+ patient + '.csv'
            path_feature_csv  = path_features + patient + '.csv'
            path_feature_xlsx = path_features + patient + '.xlsx'
            new_features(path_source,path_feature_csv,path_feature_xlsx, f_window_size, f_window_step,features)

    focal_preictal_w,focal_preictal_v  = preictal(  	patients_focal,
                                                        eig 			= True,
                                                        path_features 	= path_features,
                                                        path_csv 		= path_focal_csv,
                                                        f_window_size 	= f_window_size,
                                                        f_window_step 	= f_window_step,
                                                        eig_num 		= eig_num,
                                                        interval        = interval,
                                                        eig_window_size = eig_window_size
                                                    )
    # focal eig model
    model(X_train = list(focal_preictal_v),X_test = list(focal_interictal_v),train_data='pre',path_results = path_results,file_name = file_name)
    model(X_train = list(focal_interictal_v),X_test = list(focal_preictal_v),train_data='inter',path_results = path_results,file_name = file_name)
############################################################################
#                   EXPERIMENT WITH GENERALIZED FEATURES                   #
############################################################################
if EXP_GENERALIZED_FEATURES:
    print('EXPERIMENT WITH GENERALIZED FEATURES')
    file_name = 'EXPERIMENT WITH GENERALIZED FEATURES'
    # generalized interictal feature
    path_features   = path_gener_inter_features
    f_window_size   = feature_inter_window_size
    f_window_step   = feature_inter_window_step
    preictal_size   = feature_inter_preictal_size
    eig_window_size = feature_inter_eig_window_size
    eig_num         = feature_inter_eig_num

    if GENERALIZED_FEATURES:
        for patient in patients_gener:
            print(patient)
            path_source       = path_gener_csv + patient + '.csv'
            path_feature_csv  = path_features  + patient + '.csv'
            path_feature_xlsx = path_features  + patient + '.xlsx'
            new_features(path_source, path_feature_csv, path_feature_xlsx, f_window_size, f_window_step, features)

    gener_interictal_time, gener_interictal_feature = interictal(   patients_gener,
                                                                    eig= False,
                                                                    path_features 	=path_features,
                                                                    path_csv 		=path_gener_csv,
                                                                    preictal_size   =preictal_size,
                                                                    f_window_size 	=f_window_size,
                                                                    f_window_step   = f_window_step,
                                                                    eig_window_size = eig_window_size,
                                                                    eig_num 		= eig_num
                                                                )
    # gener preictal feature
    path_features   = path_gener_pre_features
    f_window_size   = feature_pre_window_size
    f_window_step   = feature_pre_window_step
    interval        = feature_pre_interval
    eig_window_size = feature_pre_eig_window_size
    eig_num         = feature_pre_eig_num
    if GENERALIZED_FEATURES:
        for patient in patients_gener:
            print(patient)
            path_source       = path_gener_csv+ patient + '.csv'
            path_feature_csv  = path_features + patient + '.csv'
            path_feature_xlsx = path_features + patient + '.xlsx'
            new_features(path_source, path_feature_csv, path_feature_xlsx, f_window_size, f_window_step, features)

    gener_preictal_time, gener_preictal_feature = preictal(patients_gener,
                                                           eig              = False,
                                                           path_features    = path_features,
                                                           path_csv         = path_gener_csv,
                                                           f_window_size    = f_window_size,
                                                           f_window_step    = f_window_step,
                                                           eig_num          = eig_num,
                                                           interval         = interval,
                                                           eig_window_size  = eig_window_size
                                                           )
    # gener features model
    model(X_train = list(gener_preictal_feature), X_test = list(gener_interictal_feature), train_data='pre',path_results = path_results,file_name = file_name)
    model(X_train = list(gener_interictal_feature), X_test = list(gener_preictal_feature), train_data='inter',path_results = path_results,file_name = file_name)
############################################################################
#                   EXPERIMENT WITH GENERALIZED EIG VECTOR                 #
############################################################################
if EXP_GENERALIZED_EIG:
    print('EXPERIMENT WITH GENERALIZED EIG VECTOR')
    file_name = 'EXPERIMENT WITH GENERALIZED EIG VECTOR'
    # generalized interictal eig vector
    path_features   = path_gener_inter_eig
    f_window_size   = eig_inter_window_size
    f_window_step   = eig_inter_window_step
    preictal_size   = eig_inter_preictal_size
    eig_window_size = eig_inter_eig_window_size
    eig_num         = eig_inter_eig_num

    if GENERALIZED_EIG:
        for patient in patients_gener:
            print(patient)
            path_source       = path_gener_csv + patient + '.csv'
            path_feature_csv  = path_features  + patient + '.csv'
            path_feature_xlsx = path_features  + patient + '.xlsx'
            new_features(path_source, path_feature_csv, path_feature_xlsx, f_window_size, f_window_step, features)

    gener_interictal_w, gener_interictal_v = interictal(    patients_gener,
                                                            eig= True,
                                                            path_features 	= path_features,
                                                            path_csv 		= path_gener_csv,
                                                            preictal_size   = preictal_size,
                                                            f_window_size 	= f_window_size,
                                                            f_window_step   = f_window_step,
                                                            eig_window_size = eig_window_size,
                                                            eig_num 		= eig_num
                                                       )
    # gener preictal eig vector
    path_features   = path_gener_pre_eig
    f_window_size   = eig_pre_window_size
    f_window_step   = eig_pre_window_step
    interval        = eig_pre_interval
    eig_window_size = eig_pre_eig_window_size
    eig_num         = eig_pre_eig_num 
    if GENERALIZED_EIG:
        for patient in patients_gener:
            print(patient)
            path_source       = path_gener_csv+ patient + '.csv'
            path_feature_csv = path_features + patient + '.csv'
            path_feature_xlsx = path_features + patient + '.xlsx'
            new_features(path_source, path_feature_csv, path_feature_xlsx, f_window_size, f_window_step, features)

    gener_preictal_w, gener_preictal_v = preictal(  patients_gener,
                                                    eig 		    = True,
                                                    path_features   = path_features,
                                                    path_csv 	    = path_gener_csv,
                                                    f_window_size   = f_window_size,
                                                    f_window_step   = f_window_step,
                                                    eig_num 	    = eig_num,
                                                    interval        = interval,
                                                    eig_window_size = eig_window_size
                                                 )
    # gener eig model
    model(X_train = list(gener_preictal_v),X_test = list(gener_interictal_v), train_data = 'pre',path_results = path_results,file_name = file_name)
    model(X_train = list(gener_interictal_v),X_test = list(gener_preictal_v), train_data = 'inter',path_results = path_results,file_name = file_name)
############################################################################
#                   EXPERIMENT WITH COMBINED FEATURES                      #
############################################################################
if EXP_COMBINED_FEATURES:
    print('EXPERIMENT WITH COMBINED FEATURES')
    file_name = 'EXPERIMENT WITH COMBINED FEATURES'
    if EXP_FOCAL_FEATURES and EXP_GENERALIZED_FEATURES:
        X_train = list(focal_preictal_feature)
        for i in range(gener_preictal_feature.__len__()):
            X_train.append(gener_preictal_feature[i])

        X_test = list(focal_interictal_feature)
        for i in range(gener_interictal_feature.__len__()):
            X_test.append(gener_interictal_feature[i])

        model(X_train = list(X_train),X_test = list(X_test), train_data = 'pre',path_results = path_results,file_name = file_name)
        model(X_train=list(X_test), X_test=list(X_train), train_data = 'inter', path_results=path_results, file_name=file_name)
    else:
        if not EXP_FOCAL_FEATURES:
            print("ERROR!!!\nTURN ON EXP_FOCAL_FEATURES")
        if not EXP_GENERALIZED_FEATURES:
            print("ERROR!!!\nTURN ON EXP_GENERALIZED_FEATURES")
############################################################################
#                   EXPERIMENT WITH COMBINED VECTOR                        #
############################################################################    
if EXP_COMBINED_EIG:
    print('EXPERIMENT WITH COMBINED EIG')
    file_name = 'EXPERIMENT WITH COMBINED EIG'
    if EXP_FOCAL_EIG and EXP_GENERALIZED_EIG:
        X_train = list(focal_preictal_v)
        for i in range(gener_preictal_v.__len__()):
            X_train.append(gener_preictal_v[i])

        X_test = list(focal_interictal_v)
        for i in range(gener_interictal_v.__len__()):
            X_test.append(gener_interictal_v[i])

        model(X_train=list(X_train), X_test=list(X_test), train_data = 'pre',path_results = path_results,file_name = file_name)
        model(X_train=list(X_test), X_test=list(X_train), train_data = 'inter',path_results = path_results,file_name = file_name)
    else:
        if not EXP_FOCAL_EIG:
            print("ERROR!!!\nTURN ON EXP_FOCAL_EIG")
        if not EXP_GENERALIZED_EIG:
            print("ERROR!!!\nTURN ON EXP_GENERALIZED_EIG")