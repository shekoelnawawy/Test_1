import numpy as np
import joblib
from scipy import stats
from dtaidistance import dtw, clustering
import itertools

# patients = ['540', '544', '552', '567', '584', '596', 'allsubs']
patients = ['559', '563', '570', '575', '588', '591', 'allsubs']

patient = patients[0]
df_0 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
patient = patients[1]
df_1 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
patient = patients[2]
df_2 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
patient = patients[3]
df_3 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
patient = patients[4]
df_4 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
patient = patients[5]
df_5 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)

df_0 = stats.zscore(np.array(df_0, dtype=np.double))
df_1 = stats.zscore(np.array(df_1, dtype=np.double))
df_2 = stats.zscore(np.array(df_2, dtype=np.double))
df_3 = stats.zscore(np.array(df_3, dtype=np.double))
df_4 = stats.zscore(np.array(df_4, dtype=np.double))
df_5 = stats.zscore(np.array(df_5, dtype=np.double))

numbers = ['0', '1', '2', '3', '4', '5']

timeseries = [df_0, df_1, df_2, df_3, df_4, df_5]
labels = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']

i=0
for x in itertools.permutations(numbers):
    ts = []
    lb = []
    ts.append(timeseries[int(x[0])])
    ts.append(timeseries[int(x[1])])
    ts.append(timeseries[int(x[2])])
    ts.append(timeseries[int(x[3])])
    ts.append(timeseries[int(x[4])])
    ts.append(timeseries[int(x[5])])
    lb.append(labels[int(x[0])])
    lb.append(labels[int(x[1])])
    lb.append(labels[int(x[2])])
    lb.append(labels[int(x[3])])
    lb.append(labels[int(x[4])])
    lb.append(labels[int(x[5])])

    ds = dtw.distance_matrix_fast(ts)
    print(ds)

    # You can also pass keyword arguments identical to instantiate a Hierarchical object
    model2 = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
    cluster_idx = model2.fit(ts)
    print(cluster_idx)
    # print(model2.linkage)

    model2.plot("./output_2018/hierarchy"+str(i)+".pdf", ts_label_margin = -200, show_ts_label=lb, show_tr_label=True)
    i += 1
    break


# import numpy as np
#
# train_2020 = np.load('/Users/nawawy/Desktop/ohiot1dm_data/ohiot1dm_train_2020.npy')
# test_2020 = np.load('/Users/nawawy/Desktop/ohiot1dm_data/ohiot1dm_test_2020.npy')
# train_2018 = np.load('/Users/nawawy/Desktop/ohiot1dm_data/ohiot1dm_train_2018.npy')
# test_2018 = np.load('/Users/nawawy/Desktop/ohiot1dm_data/ohiot1dm_test_2018.npy')
#
# patients = [0, 1, 2, 3, 4, 5]
# for i in patients:
#     np.save('/Users/nawawy/Desktop/ohiot1dm_data/ohiot1dm_train_2020_' + str(i) + '.npy', train_2020[train_2020[:, 0] == i, 1:])
#
#     np.save('/Users/nawawy/Desktop/ohiot1dm_data/ohiot1dm_test_2020_' + str(i) + '.npy', test_2020[test_2020[:, 0] == i, 1:])
#
#     np.save('/Users/nawawy/Desktop/ohiot1dm_data/ohiot1dm_train_2018_' + str(i) + '.npy', train_2018[train_2018[:, 0] == i, 1:])
#
#     np.save('/Users/nawawy/Desktop/ohiot1dm_data/ohiot1dm_test_2018_' + str(i) + '.npy', test_2018[test_2018[:, 0] == i, 1:])


# import joblib
# import numpy as np
# import pandas as pd
# import random
#
# # columns = ['', 'glucose', 'finger', 'basal', 'hr', 'gsr', 'carbs', 'dose', 'postprandial']
#
# year = '2020'
# patients = ['540', '544', '552', '567', '584', '596']  # , 'allsubs']
# # patients = ['559', '563', '570', '575', '588', '591'] #, 'allsubs']
# df = pd.DataFrame()
# index = 0
# # patientID, glucose, dose, finger, carbs, adversarial
#
# instances = []
#
# for patient in patients:
#     benign_data = joblib.load('/Users/nawawy/Desktop/ohio_profiling/Data/Patients/' + year + '/' + patient + '/benign_data.pkl')
#     adversarial_data = joblib.load('/Users/nawawy/Desktop/ohio_profiling/Data/Patients/' + year + '/' + patient + '/adversarial_data.pkl')
#
#     for i in range(len(adversarial_data)):
#         if adversarial_data[i][11][0] != benign_data[i][11][0]:
#             rand = random.randint(0, 1)
#             if rand % 2 == 0:
#                 instance = [index, benign_data[i][11][0], benign_data[i][11][1], benign_data[i][11][5], benign_data[i][11][2], 0]
#             else:
#                 instance = [index, adversarial_data[i][11][0], adversarial_data[i][11][1], adversarial_data[i][11][5], adversarial_data[i][11][2], 1]
#         else:
#             instance = [index, benign_data[i][11][0], benign_data[i][11][1], benign_data[i][11][5], benign_data[i][11][2], 0]
#         instances.insert(len(instances), instance)
#
#     index += 1
#
# ohiot1dm_test = np.array(instances)
#
# np.save('/Users/nawawy/Desktop/ohiot1dm_test_'+year+'.npy', ohiot1dm_test)
#
#
# df = pd.DataFrame()
#
# index = 0
# for patient in patients:
#
#     p = joblib.load('/Users/nawawy/Desktop/'+year+'data/'+patient+'.train.pkl')
#     p.insert(0, 'PatientID', index)
#     p['adversarial'] = 0
#     if not index:
#         df = p[['PatientID', 'glucose', 'dose', 'finger', 'carbs', 'adversarial']]
#     else:
#         df = pd.concat([df, p[['PatientID', 'glucose', 'dose', 'finger', 'carbs', 'adversarial']]], ignore_index=True)
#     index += 1
#
# df = df.fillna(0)
#
# train = np.array(df)
# np.save('/Users/nawawy/Desktop/ohiot1dm_train_'+year+'.npy', train)

# import numpy as np
# train = np.load('/Users/nawawy/Desktop/MAD-GANs_v2/data/data/kdd99_train.npy')
# test = np.load('/Users/nawawy/Desktop/MAD-GANs_v2/data/data/kdd99_test.npy')
#
# print(np.unique(test[:,-1]))

# import joblib
# import numpy as np
#
# patients = ['559', '563', '570', '575', '588', '591', 'allsubs']
#
#
# for patient in patients:
#     adversarial_data = joblib.load('/Data/Patients/' + patient + '/adversarial_data.pkl')
#     benign_data = joblib.load('/Data/Patients/' + patient + '/benign_data.pkl')
#     predicted_output = np.array(joblib.load('/Data/Patients/' + patient + '/predicted_output.pkl'))
#     actual_output = np.array(joblib.load('/Data/Patients/' + patient + '/actual_output.pkl'))
#
#     coefficient = np.empty([actual_output.shape[0], actual_output.shape[1]])
#     magnitude = np.empty([actual_output.shape[0], actual_output.shape[1]])
#     instantaneous_error = np.empty([actual_output.shape[0], actual_output.shape[1]])
#     for i in range(len(actual_output)):
#         postprandial = any([benign_data[i][0][7], benign_data[i][1][7], benign_data[i][2][7], benign_data[i][3][7],
#                 benign_data[i][4][7], benign_data[i][5][7], benign_data[i][6][7], benign_data[i][7][7],
#                 benign_data[i][8][7], benign_data[i][9][7], benign_data[i][10][7], benign_data[i][11][7]]) #check if postprandial (True) or fasting (False)
#         for j in range(len(actual_output[i])):
#             if not postprandial: #fasting
#                 if actual_output[i][j] < 70 and predicted_output[i][j] > 125:   # actual (hypo), predicted (hyper)
#                     coefficient[i][j] = 64
#                 elif 70 < actual_output[i][j] < 125 < predicted_output[i][j]:   # actual (normal), predicted (hyper)
#                     coefficient[i][j] = 32
#                 elif actual_output[i][j] < 70 < predicted_output[i][j] < 125:   # actual (hypo), predicted (normal)
#                     coefficient[i][j] = 16
#                 elif actual_output[i][j] > 125 and predicted_output[i][j] < 70: # actual (hyper), predicted (hypo)
#                     coefficient[i][j] = 8
#                 elif actual_output[i][j] > 125 > predicted_output[i][j] > 70:   # actual (hyper), predicted (normal)
#                     coefficient[i][j] = 4
#                 elif 125 > actual_output[i][j] > 70 > predicted_output[i][j]:   # actual (normal), predicted (hypo)
#                     coefficient[i][j] = 2
#             else:   #postprandial
#                 if actual_output[i][j] < 70 and predicted_output[i][j] > 180:   # actual (hypo), predicted (hyper)
#                     coefficient[i][j] = 64
#                 elif 70 < actual_output[i][j] < 180 < predicted_output[i][j]:   # actual (normal), predicted (hyper)
#                     coefficient[i][j] = 32
#                 elif actual_output[i][j] < 70 < predicted_output[i][j] < 180:   # actual (hypo), predicted (normal)
#                     coefficient[i][j] = 16
#                 elif actual_output[i][j] > 180 and predicted_output[i][j] < 70: # actual (hyper), predicted (hypo)
#                     coefficient[i][j] = 8
#                 elif actual_output[i][j] > 180 > predicted_output[i][j] > 70:   # actual (hyper), predicted (normal)
#                     coefficient[i][j] = 4
#                 elif 180 > actual_output[i][j] > 70 > predicted_output[i][j]:   # actual (normal), predicted (hypo)
#                     coefficient[i][j] = 2
#
#             magnitude[i][j] = pow(predicted_output[i][j] - actual_output[i][j], 2)
#             instantaneous_error[i][j] = coefficient[i][j] * magnitude[i][j]
#
#     print(instantaneous_error.shape)
#     joblib.dump(instantaneous_error,'/Data/Patients/' + patient + '/instantaneous_error.pkl')


# import joblib
# import numpy as np
#
# # patients = ['540', '544', '552', '567', '584', '596', 'allsubs']
# patients = ['559', '563', '570', '575', '588', '591', 'allsubs']
# for patient in patients:
#     f = open('/Users/nawawy/Desktop/180/'+str(patient)+'/target_benign.pkl', 'rb')
#     target_benign = np.array(joblib.load(f))
#
#     f = open('/Users/nawawy/Desktop/180/'+str(patient)+'/target_adversarial.pkl', 'rb')
#     target_adversarial = np.array(joblib.load(f))
#
#     f = open('/Users/nawawy/Desktop/180/'+str(patient)+'/median_benign.pkl', 'rb')
#     median_benign = np.array(joblib.load(f))
#
#     f = open('/Users/nawawy/Desktop/180/'+str(patient)+'/median_adversarial.pkl', 'rb')
#     median_adversarial = np.array(joblib.load(f))
#
#     target_benign = target_benign.reshape(-1)
#     target_adversarial = target_adversarial.reshape(-1)
#     median_benign = median_benign.reshape(-1)
#     median_adversarial = median_adversarial.reshape(-1)
#     print()
#     print('Model: '+str(patient))
#     print('---------------')
#     for i in range(len(target_benign)):
#         if target_benign[i] != target_adversarial[i]:
#             print(False)
#
#     target_values = 0
#     benign_predictions = 0
#     mispredictions = 0
#     print('Normal BG to Hyperglycemia')
#     for i in range(len(target_benign)):
#         if 70 < target_benign[i] < 180:
#             target_values += 1
#             if 70 < median_benign[i] < 180:
#                 benign_predictions += 1
#                 if median_adversarial[i] > 180:
#                     mispredictions += 1
#
#     print('Target values between 70 and 180 = '+str(target_values))
#     print('Benign predictions between 70 and 180 = ' + str(benign_predictions))
#     print('Mispredictions above 180 = ' + str(mispredictions))
#     print('percentage mispredictions (mispredictions/benign): ' + str((mispredictions / benign_predictions) * 100))
#     print()
#
#     target_values = 0
#     benign_predictions = 0
#     mispredictions = 0
#     print('Hypoglycemia to Hyperglycemia')
#     for i in range(len(target_benign)):
#         if target_benign[i] < 70:
#             target_values += 1
#             if median_benign[i] < 70:
#                 benign_predictions += 1
#                 if median_adversarial[i] > 180:
#                     mispredictions += 1
#
#     print('Target values less than 70 = '+str(target_values))
#     print('Benign predictions less than 70 = ' + str(benign_predictions))
#     print('Mispredictions above 180 = ' + str(mispredictions))
#     print('percentage mispredictions (mispredictions/benign): ' + str((mispredictions / benign_predictions) * 100))
#     print()


# import joblib
# import numpy as np
#
# # patients = ['540', '544', '552', '567', '584', '596', 'allsubs']
# patients = ['559', '563', '570', '575', '588', '591', 'allsubs']
# for patient in patients:
#     f = open('/Users/nawawy/Desktop/130/'+str(patient)+'/target_benign.pkl', 'rb')
#     target_benign = np.array(joblib.load(f))
#
#     f = open('/Users/nawawy/Desktop/130/'+str(patient)+'/target_adversarial.pkl', 'rb')
#     target_adversarial = np.array(joblib.load(f))
#
#     f = open('/Users/nawawy/Desktop/130/'+str(patient)+'/median_benign.pkl', 'rb')
#     median_benign = np.array(joblib.load(f))
#
#     f = open('/Users/nawawy/Desktop/130/'+str(patient)+'/median_adversarial.pkl', 'rb')
#     median_adversarial = np.array(joblib.load(f))
#
#     target_benign = target_benign.reshape(-1)
#     target_adversarial = target_adversarial.reshape(-1)
#     median_benign = median_benign.reshape(-1)
#     median_adversarial = median_adversarial.reshape(-1)
#     print()
#     print('Model: '+str(patient))
#     print('---------------')
#     for i in range(len(target_benign)):
#         if target_benign[i] != target_adversarial[i]:
#             print(False)
#
#     target_values = 0
#     benign_predictions = 0
#     mispredictions = 0
#     print('Normal BG to Hyperglycemia')
#     for i in range(len(target_benign)):
#         if 70 < target_benign[i] < 130:
#             target_values += 1
#             if 70 < median_benign[i] < 130:
#                 benign_predictions += 1
#                 if median_adversarial[i] > 130:
#                     mispredictions += 1
#
#     print('Target values between 70 and 130 = '+str(target_values))
#     print('Benign predictions between 70 and 130 = ' + str(benign_predictions))
#     print('Mispredictions above 130 = ' + str(mispredictions))
#     print('percentage mispredictions (mispredictions/benign): ' + str((mispredictions / benign_predictions) * 100))
#     print()
#
#     target_values = 0
#     benign_predictions = 0
#     mispredictions = 0
#     print('Hypoglycemia to Hyperglycemia')
#     for i in range(len(target_benign)):
#         if target_benign[i] < 70:
#             target_values += 1
#             if median_benign[i] < 70:
#                 benign_predictions += 1
#                 if median_adversarial[i] > 130:
#                     mispredictions += 1
#
#     print('Target values less than 70 = '+str(target_values))
#     print('Benign predictions less than 70 = ' + str(benign_predictions))
#     print('Mispredictions above 130 = ' + str(mispredictions))
#     print('percentage mispredictions (mispredictions/benign): ' + str((mispredictions / benign_predictions) * 100))
#     print()

# import joblib
# import pandas as pd
# import joblib
# import numpy as np
# from sklearn import tree
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# from tsfresh import select_features
#
# feature_set = 'Comprehensive' #'Minimal
#
# features_2018 = joblib.load('/Users/nawawy/Desktop/'+feature_set+'_2018.pkl')
# features_2020 = joblib.load('/Users/nawawy/Desktop/'+feature_set+'_2020.pkl')
#
# df = pd.concat([features_2018, features_2020], ignore_index=True)
#
# standardize = False
# features = df # joblib.load('/Users/nawawy/Desktop/feature_engineering.pkl')
# features.dropna(axis=1, inplace=True)
# # print(features)
# y = pd.Series(['High', 'High', 'High', 'High', 'High', 'Low', 'High', 'Low', 'Low', 'High', 'High', 'High'])
# # print(y)
# features_filtered = select_features(features, y)
#
# print(features_filtered)
# exit(1)
# # Using DataFrame.insert() to add a column
# features.insert(len(features.columns), "Vulnerability", ['High', 'High', 'High', 'High', 'High', 'Low', 'High', 'Low', 'Low', 'High', 'High', 'High'], True)
# column_names = features.columns
# if standardize:
#     data = StandardScaler().fit_transform(features.iloc[:, :-1])
# else:
#     data = np.array(features.iloc[:,:-1])
#
# vulnerability = np.array(features.iloc[:,-1])
#
#
#
# seeds = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000]
# for seed in seeds:
#     clf = tree.DecisionTreeClassifier(random_state=seed)
#     clf = clf.fit(data, vulnerability)
#     tree.plot_tree(clf, feature_names=column_names[:-1], filled=True, class_names=['High', 'Low'], rounded = True)
#     # plt.show()
#     plt.savefig('./CombinedTrees/tree_'+str(seed), dpi=300)


# print(features_2020.columns)
# print(features_2018['finger__ar'])
# print(features_2020['glucose__fft_coefficient__attr_"abs"__coeff_78'])
#
# print('-----------------------------------')
#
# print(features_2018['glucose__minimum'])
# print(features_2020['glucose__minimum'])


# import numpy as np
# import joblib
# from scipy import stats
# from dtaidistance import dtw, clustering
# import itertools
#
# patients = ['559', '563', '570', '575', '588', '591', 'allsubs']
#
# patient = patients[0]
# df_0 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[1]
# df_1 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[2]
# df_2 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[3]
# df_3 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[4]
# df_4 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[5]
# df_5 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
#
# df_0 = stats.zscore(np.array(df_0, dtype=np.double))
# df_1 = stats.zscore(np.array(df_1, dtype=np.double))
# df_2 = stats.zscore(np.array(df_2, dtype=np.double))
# df_3 = stats.zscore(np.array(df_3, dtype=np.double))
# df_4 = stats.zscore(np.array(df_4, dtype=np.double))
# df_5 = stats.zscore(np.array(df_5, dtype=np.double))
#
# numbers = ['0', '1', '2', '3', '4', '5']
#
# timeseries = [df_0, df_1, df_2, df_3, df_4, df_5]
# labels = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
#
# i=0
# for x in itertools.permutations(numbers):
#     ts = []
#     lb = []
#     ts.append(timeseries[int(x[0])])
#     ts.append(timeseries[int(x[1])])
#     ts.append(timeseries[int(x[2])])
#     ts.append(timeseries[int(x[3])])
#     ts.append(timeseries[int(x[4])])
#     ts.append(timeseries[int(x[5])])
#     lb.append(labels[int(x[0])])
#     lb.append(labels[int(x[1])])
#     lb.append(labels[int(x[2])])
#     lb.append(labels[int(x[3])])
#     lb.append(labels[int(x[4])])
#     lb.append(labels[int(x[5])])
#
#     ds = dtw.distance_matrix_fast(ts)
#     #print(ds)
#
#     # You can also pass keyword arguments identical to instantiate a Hierarchical object
#     model2 = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
#     cluster_idx = model2.fit(ts)
#     # print(cluster_idx)
#     # print(model2.linkage)
#
#     model2.plot("./output_2018/hierarchy"+str(i)+".pdf", ts_label_margin = -200, show_ts_label=lb)
#     i += 1


# from dtaidistance import dtw_ndim
# import numpy as np
# import joblib
# from scipy import stats
# from dtaidistance import dtw, clustering
# import math
# import random
#
# seeds = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000, 2000, 3000, 4000, 5000, 10000, 100000]
# patients = ['559', '563', '570', '575', '588', '591', 'allsubs']
#
# for s in seeds:
#     random.seed(s)
#     patient = patients[0]
#     df_0 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
#     patient = patients[1]
#     df_1 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
#     patient = patients[2]
#     df_2 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
#     patient = patients[3]
#     df_3 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
#     patient = patients[4]
#     df_4 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
#     patient = patients[5]
#     df_5 = joblib.load('./Data/'+patient+'/instantaneous_error.pkl').mean(axis=1)
#
#     df_0 = stats.zscore(np.array(df_0, dtype=np.double))
#     df_1 = stats.zscore(np.array(df_1, dtype=np.double))
#     df_2 = stats.zscore(np.array(df_2, dtype=np.double))
#     df_3 = stats.zscore(np.array(df_3, dtype=np.double))
#     df_4 = stats.zscore(np.array(df_4, dtype=np.double))
#     df_5 = stats.zscore(np.array(df_5, dtype=np.double))
#
#
#     # d = dtw_ndim.distance(df_0, df_1)
#     # print(d)
#
#     timeseries = [df_1, df_2, df_3, df_4, df_5, df_0]
#
#     ds = dtw.distance_matrix(timeseries)
#     # print(ds)
#
#
#     # You can also pass keyword arguments identical to instantiate a Hierarchical object
#     model2 = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
#     cluster_idx = model2.fit(timeseries)
#
#     # print(model2.linkage)
#
#     model2.plot("./output_slow/hierarchy_"+str(s)+".pdf")



# import joblib
# import numpy as np
# from sklearn import tree
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
#
# standardize = False
# features = joblib.load('/Users/nawawy/Desktop/feature_engineering_comprehensive.pkl')
#
# # Using DataFrame.insert() to add a column
# features.insert(len(features.columns), "Vulnerability", ['High', 'Low', 'Low', 'High', 'High', 'High'], True)
# column_names = features.columns
#
# if standardize:
#     data = StandardScaler().fit_transform(features.iloc[:, :-1])
# else:
#     data = np.array(features.iloc[:,:-1])
#
# vulnerability = np.array(features.iloc[:,-1])
#
#
#
# # seeds = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000]
# for seed in range(100):
#     clf = tree.DecisionTreeClassifier(random_state=seed)
#     clf = clf.fit(data, vulnerability)
#     tree.plot_tree(clf, feature_names=column_names[:-1], filled=True, class_names=['High', 'Low'], rounded = True)
#     # plt.show()
#     plt.savefig('./ComprehensiveTrees/tree_'+str(seed), dpi=300 )


    # dot_data = tree.export_graphviz(clf, out_file='./Trees/tree_'+str(seed), feature_names = column_names[:-1],
    # class_names = ['High', 'Low'], filled = True, rounded = True, special_characters = True)
    # graph = graphviz.Source('./Trees/tree_'+str(seed))
    # graph.render("tree_")


# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")


# import numpy as np
# import joblib
# from scipy import stats
# from dtaidistance import dtw, clustering
# import itertools
#
# patients = ['540', '544', '552', '567', '584', '596', 'allsubs']
#
# patient = patients[0]
# df_0 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[1]
# df_1 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[2]
# df_2 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[3]
# df_3 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[4]
# df_4 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[5]
# df_5 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
#
# df_0 = stats.zscore(np.array(df_0, dtype=np.double))
# df_1 = stats.zscore(np.array(df_1, dtype=np.double))
# df_2 = stats.zscore(np.array(df_2, dtype=np.double))
# df_3 = stats.zscore(np.array(df_3, dtype=np.double))
# df_4 = stats.zscore(np.array(df_4, dtype=np.double))
# df_5 = stats.zscore(np.array(df_5, dtype=np.double))
#
# numbers = ['0', '1', '2', '3', '4', '5']
#
# timeseries = [df_0, df_1, df_2, df_3, df_4, df_5]
# labels = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
#
# i=0
# for x in itertools.permutations(numbers):
#     ts = []
#     lb = []
#     ts.append(timeseries[int(x[0])])
#     ts.append(timeseries[int(x[1])])
#     ts.append(timeseries[int(x[2])])
#     ts.append(timeseries[int(x[3])])
#     ts.append(timeseries[int(x[4])])
#     ts.append(timeseries[int(x[5])])
#     lb.append(labels[int(x[0])])
#     lb.append(labels[int(x[1])])
#     lb.append(labels[int(x[2])])
#     lb.append(labels[int(x[3])])
#     lb.append(labels[int(x[4])])
#     lb.append(labels[int(x[5])])
#
#     ds = dtw.distance_matrix_fast(ts)
#     #print(ds)
#
#     # You can also pass keyword arguments identical to instantiate a Hierarchical object
#     model2 = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
#     cluster_idx = model2.fit(ts)
#     print(cluster_idx)
#     print(model2.linkage)
#
#     model2.plot("./output/hierarchy"+str(i)+".pdf", ts_label_margin = -200, show_ts_label=lb)
#     i += 1

#













# from dtaidistance import dtw_ndim
# import numpy as np
# import joblib
# from scipy import stats
# from dtaidistance import dtw, clustering
# import math
# import random
#
# patients = ['540', '544', '552', '567', '584', '596', 'allsubs']
#
# patient = patients[0]
# df_0 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[1]
# df_1 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[2]
# df_2 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[3]
# df_3 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[4]
# df_4 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[5]
# df_5 = joblib.load('./180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
#
# df_0 = stats.zscore(np.array(df_0, dtype=np.double))
# df_1 = stats.zscore(np.array(df_1, dtype=np.double))
# df_2 = stats.zscore(np.array(df_2, dtype=np.double))
# df_3 = stats.zscore(np.array(df_3, dtype=np.double))
# df_4 = stats.zscore(np.array(df_4, dtype=np.double))
# df_5 = stats.zscore(np.array(df_5, dtype=np.double))
#
#
# timeseries = [df_0, df_1, df_2, df_3, df_4, df_5]
# ds = dtw.distance_matrix_fast(timeseries)
# # You can also pass keyword arguments identical to instantiate a Hierarchical object
# model2 = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
# cluster_idx = model2.fit(timeseries)
# model2.plot("hierarchy_1.pdf")

# import joblib
# import pandas as pd
# import math
# from tsfresh import extract_features
# from sklearn.preprocessing import StandardScaler
#
#
# features = ['', 'glucose', 'finger', 'carbs', 'dose']
#
# patients = ['540', '544', '552', '567', '584', '596', 'allsubs']
# split = 'train'
#
# patient = patients[0]
# df_0 = joblib.load('/Users/nawawy/Desktop/data/'+patient+'.'+split+'.pkl')
# patient = patients[1]
# df_1 = joblib.load('/Users/nawawy/Desktop/data/'+patient+'.'+split+'.pkl')
# patient = patients[2]
# df_2 = joblib.load('/Users/nawawy/Desktop/data/'+patient+'.'+split+'.pkl')
# patient = patients[3]
# df_3 = joblib.load('/Users/nawawy/Desktop/data/'+patient+'.'+split+'.pkl')
# patient = patients[4]
# df_4 = joblib.load('/Users/nawawy/Desktop/data/'+patient+'.'+split+'.pkl')
# patient = patients[5]
# df_5 = joblib.load('/Users/nawawy/Desktop/data/'+patient+'.'+split+'.pkl')
#
# df_0.finger.fillna(df_0.glucose, inplace=True)
# df_0.glucose.fillna(df_0.finger, inplace=True)
#
# df_1.finger.fillna(df_1.glucose, inplace=True)
# df_1.glucose.fillna(df_1.finger, inplace=True)
#
# df_2.finger.fillna(df_2.glucose, inplace=True)
# df_2.glucose.fillna(df_2.finger, inplace=True)
#
# df_3.finger.fillna(df_3.glucose, inplace=True)
# df_3.glucose.fillna(df_3.finger, inplace=True)
#
# df_4.finger.fillna(df_4.glucose, inplace=True)
# df_4.glucose.fillna(df_4.finger, inplace=True)
#
# df_5.finger.fillna(df_5.glucose, inplace=True)
# df_5.glucose.fillna(df_5.finger, inplace=True)
#
# df_0['carbs'][0] = 0 if math.isnan(df_0['carbs'][0]) else df_0['carbs'][0]
# df_1['carbs'][0] = 0 if math.isnan(df_1['carbs'][0]) else df_1['carbs'][0]
# df_2['carbs'][0] = 0 if math.isnan(df_2['carbs'][0]) else df_2['carbs'][0]
# df_3['carbs'][0] = 0 if math.isnan(df_3['carbs'][0]) else df_3['carbs'][0]
# df_4['carbs'][0] = 0 if math.isnan(df_4['carbs'][0]) else df_4['carbs'][0]
# df_5['carbs'][0] = 0 if math.isnan(df_5['carbs'][0]) else df_5['carbs'][0]
#
# df_0['dose'][0] = 0 if math.isnan(df_0['dose'][0]) else df_0['dose'][0]
# df_1['dose'][0] = 0 if math.isnan(df_1['dose'][0]) else df_1['dose'][0]
# df_2['dose'][0] = 0 if math.isnan(df_2['dose'][0]) else df_2['dose'][0]
# df_3['dose'][0] = 0 if math.isnan(df_3['dose'][0]) else df_3['dose'][0]
# df_4['dose'][0] = 0 if math.isnan(df_4['dose'][0]) else df_4['dose'][0]
# df_5['dose'][0] = 0 if math.isnan(df_5['dose'][0]) else df_5['dose'][0]
#
# df_0 = df_0[features].ffill()
# df_1 = df_1[features].ffill()
# df_2 = df_2[features].ffill()
# df_3 = df_3[features].ffill()
# df_4 = df_4[features].ffill()
# df_5 = df_5[features].ffill()
#
#
# df_0['PatientID'] = 0
# df_1['PatientID'] = 1
# df_2['PatientID'] = 2
# df_3['PatientID'] = 3
# df_4['PatientID'] = 4
# df_5['PatientID'] = 5
#
# df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5])
# # df = [df_0, df_1, df_2, df_3, df_4, df_5]
# # print(df)
# extracted_features = extract_features(df, column_id='PatientID', column_sort='') #, default_fc_parameters=MinimalFCParameters())
# print(extracted_features)
#
# extracted_features.dropna(axis=1, inplace=True)
# print(extracted_features)
#
# #create scaled DataFrame where each variable has mean of 0 and standard dev of 1
# scaled_df = StandardScaler().fit_transform(extracted_features)
# print(scaled_df)
# joblib.dump(scaled_df, './feature_engineering.pkl')


# from dtaidistance import dtw_ndim
# import numpy as np
# import joblib
# from scipy import stats
# from dtaidistance import dtw, clustering
# import math
# import matplotlib.pyplot as pp
# # features = ['glucose', 'finger', 'carbs', 'dose']
# # feature = features[0]
#
# patients = ['540', '544', '552', '567', '584', '596', 'allsubs']
#
# patient = patients[0]
# df_0 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[1]
# df_1 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[2]
# df_2 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[3]
# df_3 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[4]
# df_4 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
# patient = patients[5]
# df_5 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/180/'+patient+'/instantaneous_error.pkl').mean(axis=1)
#
# pp.plot(np.arange(len(df_5)),df_5)
# pp.show()
# df_0 = stats.zscore(np.array(df_0, dtype=np.double))
# df_1 = stats.zscore(np.array(df_1, dtype=np.double))
# df_2 = stats.zscore(np.array(df_2, dtype=np.double))
# df_3 = stats.zscore(np.array(df_3, dtype=np.double))
# df_4 = stats.zscore(np.array(df_4, dtype=np.double))
# df_5 = stats.zscore(np.array(df_5, dtype=np.double))
#
#
# # d = dtw_ndim.distance(df_0, df_1)
# # print(d)
#
# timeseries = [df_0, df_1, df_2, df_3, df_4, df_5]
#
# ds = dtw.distance_matrix_fast(timeseries)
# print(ds)
#
#
# # You can also pass keyword arguments identical to instantiate a Hierarchical object
# model2 = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
# cluster_idx = model2.fit(timeseries)
#
# print(model2.linkage)
#
# model2.plot("hierarchy.pdf")


# SciPy linkage clustering
# model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})
# cluster_idx = model3.fit(timeseries)
# print(model3.linkage)
# model3.plot("hierarchy.png")




# import joblib
# from tslearn.clustering import TimeSeriesKMeans
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# import numpy
#
# patients = ['540', '544', '552', '567', '584', '596', 'allsubs']
# threshold = 130 # 130 fasting / 180 postprandial
#
#
# patient = patients[0]
# df_0 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/'+str(threshold)+'/'+patient+'/instantaneous_error.pkl')
# patient = patients[1]
# df_1 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/'+str(threshold)+'/'+patient+'/instantaneous_error.pkl')
# patient = patients[2]
# df_2 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/'+str(threshold)+'/'+patient+'/instantaneous_error.pkl')
# patient = patients[3]
# df_3 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/'+str(threshold)+'/'+patient+'/instantaneous_error.pkl')
# patient = patients[4]
# df_4 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/'+str(threshold)+'/'+patient+'/instantaneous_error.pkl')
# patient = patients[5]
# df_5 = joblib.load('/Users/nawawy/Desktop/InstantaneousErrors/'+str(threshold)+'/'+patient+'/instantaneous_error.pkl')
#
# print(df_0.shape)
# print(df_1.shape)
# print(df_2.shape)
# print(df_3.shape)
# print(df_4.shape)
# print(df_5.shape)
# exit(1)
#
#
# size = min(len(df_0), len(df_1), len(df_2), len(df_3), len(df_4), len(df_5))
# # df = numpy.array([df_0.to_numpy()[:size,:], df_1.to_numpy()[:size,:], df_2.to_numpy()[:size,:], df_3.to_numpy()[:size,:], df_4.to_numpy()[:size,:], df_5.to_numpy()[:size,:]])
# df = numpy.array([df_0[:size], df_1[:size], df_2[:size], df_3[:size], df_4[:size], df_5[:size]])
# # numpy.random.shuffle(df)
# # print(df)
#
# # Keep only 50 time series
# X_train = TimeSeriesScalerMeanVariance().fit_transform(df) #df[:50]
# # Make time series shorter
# # X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
# sz = X_train.shape[1]
#
#
# seeds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
# for seed in seeds:
#     numpy.random.seed(seed)
#     # DBA-k-means
#     # print("DBA k-means")
#     dba_km = TimeSeriesKMeans(n_clusters=2,
#                               n_init=10,
#                               metric="dtw",
#                               verbose=False,
#                               max_iter_barycenter=10,
#                               random_state=seed)
#     y_pred = dba_km.fit_predict(X_train)
#
#     print('seed:' + str(seed))
#     print('labels:')
#     print(dba_km.labels_)
#     print('------------------------------------------')
#     print('inertia:')
#     print(dba_km.inertia_)
#     print('------------------------------------------')
#
#
#



#----------------------------------------------------------------------------------------------------
# indices = []
# for i in range(len(benign_data)):
#     if any([benign_data[i][7], benign_data[i][15], benign_data[i][23], benign_data[i][31],
#             benign_data[i][39], benign_data[i][47], benign_data[i][55], benign_data[i][63],
#             benign_data[i][71], benign_data[i][79], benign_data[i][87], benign_data[i][95]]):
#         indices.append(i)
#
# arr = np.array(indices)
# print(arr.shape)
#
# print(benign[arr].shape)

# import joblib
# import pandas as pd
# import numpy as np
#
# patients = ['540', '544', '552', '567', '584', '596']#, 'allsubs']
# patient = patients[0]
# threshold = 180 # 130 fasting / 180 postprandial
#
# # df_adv = pd.DataFrame()
# # df_ben = pd.DataFrame()
# #
# # for patient in patients:
# #     adversarial_data = joblib.load('/Users/nawawy/Desktop/Results/Summary/'+str(threshold)+'/'+patient+'/adversarial_data.pkl')
# #     benign_data = joblib.load('/Users/nawawy/Desktop/Results/Summary/'+str(threshold)+'/'+patient+'/benign_data.pkl')
#     # predicted_output = numpy.array(joblib.load('/Users/nawawy/Desktop/Results/Summary/'+str(threshold)+'/'+patient+'/predicted_output.pkl'))
#     # actual_output = numpy.array(joblib.load('/Users/nawawy/Desktop/Results/Summary/'+str(threshold)+'/'+patient+'/actual_output.pkl'))
#
#     # adversarial_data = adversarial_data.reshape(len(adversarial_data), -1)
#     # benign_data = benign_data.reshape(len(benign_data), -1)
#
#     # df_a = pd.DataFrame(adversarial_data, columns=['glucose (t-12)', 'dose (t-12)', 'carbs (t-12)', 't1 (t-12)', 't2 (t-12)', 'finger (t-12)', 'missing (t-12)', 'glucose (t-11)', 'dose (t-11)', 'carbs (t-11)', 't1 (t-11)', 't2 (t-11)', 'finger (t-11)', 'missing (t-11)', 'glucose (t-10)', 'dose (t-10)', 'carbs (t-10)', 't1 (t-10)', 't2 (t-10)', 'finger (t-10)', 'missing (t-10)', 'glucose (t-9)', 'dose (t-9)', 'carbs (t-9)', 't1 (t-9)', 't2 (t-9)', 'finger (t-9)', 'missing (t-9)', 'glucose (t-8)', 'dose (t-8)', 'carbs (t-8)', 't1 (t-8)', 't2 (t-8)', 'finger (t-8)', 'missing (t-8)', 'glucose (t-7)', 'dose (t-7)', 'carbs (t-7)', 't1 (t-7)', 't2 (t-7)', 'finger (t-7)', 'missing (t-7)', 'glucose (t-6)', 'dose (t-6)', 'carbs (t-6)', 't1 (t-6)', 't2 (t-6)', 'finger (t-6)', 'missing (t-6)', 'glucose (t-5)', 'dose (t-5)', 'carbs (t-5)', 't1 (t-5)', 't2 (t-5)', 'finger (t-5)', 'missing (t-5)', 'glucose (t-4)', 'dose (t-4)', 'carbs (t-4)', 't1 (t-4)', 't2 (t-4)', 'finger (t-4)', 'missing (t-4)', 'glucose (t-3)', 'dose (t-3)', 'carbs (t-3)', 't1 (t-3)', 't2 (t-3)', 'finger (t-3)', 'missing (t-3)', 'glucose (t-2)', 'dose (t-2)', 'carbs (t-2)', 't1 (t-2)', 't2 (t-2)', 'finger (t-2)', 'missing (t-2)', 'glucose (t-1)', 'dose (t-1)', 'carbs (t-1)', 't1 (t-1)', 't2 (t-1)', 'finger (t-1)', 'missing (t-1)'])
#     # df_b = pd.DataFrame(benign_data, columns=['glucose (t-12)', 'dose (t-12)', 'carbs (t-12)', 't1 (t-12)', 't2 (t-12)', 'finger (t-12)', 'missing (t-12)', 'glucose (t-11)', 'dose (t-11)', 'carbs (t-11)', 't1 (t-11)', 't2 (t-11)', 'finger (t-11)', 'missing (t-11)', 'glucose (t-10)', 'dose (t-10)', 'carbs (t-10)', 't1 (t-10)', 't2 (t-10)', 'finger (t-10)', 'missing (t-10)', 'glucose (t-9)', 'dose (t-9)', 'carbs (t-9)', 't1 (t-9)', 't2 (t-9)', 'finger (t-9)', 'missing (t-9)', 'glucose (t-8)', 'dose (t-8)', 'carbs (t-8)', 't1 (t-8)', 't2 (t-8)', 'finger (t-8)', 'missing (t-8)', 'glucose (t-7)', 'dose (t-7)', 'carbs (t-7)', 't1 (t-7)', 't2 (t-7)', 'finger (t-7)', 'missing (t-7)', 'glucose (t-6)', 'dose (t-6)', 'carbs (t-6)', 't1 (t-6)', 't2 (t-6)', 'finger (t-6)', 'missing (t-6)', 'glucose (t-5)', 'dose (t-5)', 'carbs (t-5)', 't1 (t-5)', 't2 (t-5)', 'finger (t-5)', 'missing (t-5)', 'glucose (t-4)', 'dose (t-4)', 'carbs (t-4)', 't1 (t-4)', 't2 (t-4)', 'finger (t-4)', 'missing (t-4)', 'glucose (t-3)', 'dose (t-3)', 'carbs (t-3)', 't1 (t-3)', 't2 (t-3)', 'finger (t-3)', 'missing (t-3)', 'glucose (t-2)', 'dose (t-2)', 'carbs (t-2)', 't1 (t-2)', 't2 (t-2)', 'finger (t-2)', 'missing (t-2)', 'glucose (t-1)', 'dose (t-1)', 'carbs (t-1)', 't1 (t-1)', 't2 (t-1)', 'finger (t-1)', 'missing (t-1)'])
#
#
#     # df_adv = pd.concat([df_adv, df_a])
#     # df_ben = pd.concat([df_ben, df_b])
#
# # print(df_ben)
#
# benign_data = joblib.load('/Users/nawawy/Desktop/Results/Summary/' + str(threshold) + '/' + patient + '/benign_data.pkl')
# # print(benign_data)
# benign_data = benign_data.reshape(len(benign_data), -1)
# print(benign_data.shape)
# df_b = pd.DataFrame(benign_data, columns=['glucose (t-12)', 'dose (t-12)', 'carbs (t-12)', 't1 (t-12)', 't2 (t-12)', 'finger (t-12)', 'missing (t-12)', 'glucose (t-11)', 'dose (t-11)', 'carbs (t-11)', 't1 (t-11)', 't2 (t-11)', 'finger (t-11)', 'missing (t-11)', 'glucose (t-10)', 'dose (t-10)', 'carbs (t-10)', 't1 (t-10)', 't2 (t-10)', 'finger (t-10)', 'missing (t-10)', 'glucose (t-9)', 'dose (t-9)', 'carbs (t-9)', 't1 (t-9)', 't2 (t-9)', 'finger (t-9)', 'missing (t-9)', 'glucose (t-8)', 'dose (t-8)', 'carbs (t-8)', 't1 (t-8)', 't2 (t-8)', 'finger (t-8)', 'missing (t-8)', 'glucose (t-7)', 'dose (t-7)', 'carbs (t-7)', 't1 (t-7)', 't2 (t-7)', 'finger (t-7)', 'missing (t-7)', 'glucose (t-6)', 'dose (t-6)', 'carbs (t-6)', 't1 (t-6)', 't2 (t-6)', 'finger (t-6)', 'missing (t-6)', 'glucose (t-5)', 'dose (t-5)', 'carbs (t-5)', 't1 (t-5)', 't2 (t-5)', 'finger (t-5)', 'missing (t-5)', 'glucose (t-4)', 'dose (t-4)', 'carbs (t-4)', 't1 (t-4)', 't2 (t-4)', 'finger (t-4)', 'missing (t-4)', 'glucose (t-3)', 'dose (t-3)', 'carbs (t-3)', 't1 (t-3)', 't2 (t-3)', 'finger (t-3)', 'missing (t-3)', 'glucose (t-2)', 'dose (t-2)', 'carbs (t-2)', 't1 (t-2)', 't2 (t-2)', 'finger (t-2)', 'missing (t-2)', 'glucose (t-1)', 'dose (t-1)', 'carbs (t-1)', 't1 (t-1)', 't2 (t-1)', 'finger (t-1)', 'missing (t-1)'])
# for i in range(1, 13):
#     df_b.insert(7*i+i-1, "postprandial (t-"+str(13-i)+")", np.full(len(benign_data), False))
#
# for i in range(0, len(df_b)):
#     if(df_b["carbs (t-1)"].iloc[0] > 0):
#         print(True)
