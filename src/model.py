'''
GNU GENERAL PUBLIC LICENSE
Version 1, Dec 2018

Copyright (C) 2018, Meta_Learners
xiongz17@mails.tsinghua.edu.cn
jiangjy17@mails.tsinghua.edu.cn
zhangwenpeng0@gmail.com

Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.
'''

'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import os
os.system('pip3 install lightgbm==2.1.2')
os.system('pip3 install hyperopt')

import pandas as pd
import pickle
import data_converter
import numpy as np
import scipy
from os.path import isfile
import random
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import sklearn
from multiprocessing import Pool

from preprocess import *
from boosting import *
from automl import *

module = ['read_data', 'preprocess', 'encode_cat', 'encode_mv', 'fit', 'predict']

class Model:
    def __init__(self, data_info, time_info):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples = 0
        self.num_feat = 1
        self.num_labels = 1
        
        self.is_trained = False
        self.batch_num = 0
        self.total_batch_num = 10
        
        self.batch_window_size = 3

        self.y_pred = None
        self.data_memory = []

        self.hyper_param = None

        self.missing_value_preprocess = MissingValuePreprocess(drop_ratio=0.9)

        self.overall_time = np.zeros(10, dtype='float')
        self.module_time = {}
        for m in module:
            self.module_time[m] = np.zeros(10, dtype='float')


    def fit(self, F, y, data_info, time_info):

        # time budget score
        if not self.is_trained:
            weight = [1., 1., 2., 4.]
            feat_weight = 0.0
            for i in range(4):
                feat_weight += weight[i] * data_info['loaded_feat_types'][i]
            self.budget_score = float(data_info['time_budget']) / (y.shape[0] * 10. / 1e6) / feat_weight
            self.time_manager = TimeManager(self.budget_score, self.batch_window_size)
            print ('budget score: %.2f' %(self.budget_score))

            self.use_mv = False
            if (data_info['loaded_feat_types'][3] != 0 and self.budget_score > 3.5):
                self.use_mv = True

        # read data
        if not self.is_trained:
            self.module_time['read_data'][self.batch_num] = time.time() - time_info[1]

        y = y.ravel()
        
        # preprocessing
        if not self.is_trained:

            module_start = time.time()
            
            F['numerical'] = self.missing_value_preprocess.fit_transform(F['numerical'], 'numerical', input_type='ndarray')
            F['CAT'] = self.missing_value_preprocess.fit_transform(F['CAT'], 'CAT', input_type='dataframe')
            if self.use_mv:
                F['MV'] = self.missing_value_preprocess.fit_transform(F['MV'], 'MV', input_type='dataframe')

            F['numerical'] = data_converter.replace_missing(F['numerical']).astype('float32')
            F['CAT'] = F['CAT'].fillna('-1')

            module_end = time.time()
            self.module_time['preprocess'][self.batch_num] = module_end - module_start

            self.F = F

        # store current batch of data
        if (len(self.data_memory) == self.batch_window_size):
            del self.data_memory[0]
        self.data_memory.append([self.F, y])

        self.batch_end_time = time.time()
        if self.is_trained:
            self.batch_start_time = self.next_batch_start_time
        else:
            self.batch_start_time = time_info[1]
        
        self.overall_time[self.batch_num] = self.batch_end_time - self.batch_start_time
        print ('overall time spent on batch %d: %.2f seconds' %(self.batch_num, self.overall_time[self.batch_num]))
        for m in module:
            t = self.module_time[m][self.batch_num]
            ratio = t / self.overall_time[self.batch_num]
            print ('%s: %.2f seconds, %.2f%%' %(m, t, ratio * 100.))
        if self.is_trained:
            print ('time spent ratio: %.2f%%' %(self.time_spent_ratio))

        self.fit_end_time = time.time()
        self.next_batch_start_time = time.time()


    def transferPredict(self, i, data_info, time_info):

        train_num = np.concatenate([self.data_memory[j][0]['numerical'] for j in range(i, len(self.data_memory))], axis=0)
        train_cat = pd.concat([self.data_memory[j][0]['CAT'] for j in range(i, len(self.data_memory))], axis=0, ignore_index=True, copy=False)
        label = np.concatenate([self.data_memory[j][1] for j in range(i, len(self.data_memory))])

        test_num = self.F['numerical']
        test_cat = self.F['CAT']

        module_start = time.time()

        # encode categorical feature
        all_cat = pd.concat([train_cat, test_cat], axis=0, ignore_index=True, copy=False)
        del train_cat, test_cat
        cat_columns = all_cat.columns.copy()
        
        with Pool(processes=2) as pool:
            cat_encode = pool.map(CATEncoder().fit_transform, [all_cat[[col]] for col in cat_columns])
            pool.close()
            pool.join()
        all_cat = pd.concat(cat_encode, axis=1, copy=False)
        del cat_encode

        train_cat, test_cat = all_cat.iloc[:label.shape[0], :].reset_index(drop=True), all_cat.iloc[label.shape[0]:, :].reset_index(drop=True)
        del all_cat

        module_end = time.time()
        self.module_time['encode_cat'][self.batch_num] = module_end - module_start

        train_num, test_num = pd.DataFrame(train_num), pd.DataFrame(test_num)
        train_feature = [train_num, train_cat]
        test_feature = [test_num, test_cat]
        
        # encode multi-value feature
        if self.use_mv:

            module_start = time.time()

            max_cat_num = 1000

            all_mv = pd.concat([self.data_memory[j][0]['MV'] for j in range(i, len(self.data_memory))], axis=0)
            all_mv = pd.concat([all_mv, self.F['MV']], axis=0)
            mv_columns = all_mv.columns.copy()

            with Pool(processes=2) as pool:
                mv_encode = pool.map(MVEncoder(max_cat_num=max_cat_num).fit_transform, [all_mv[col] for col in mv_columns])
                all_mv = pd.concat(mv_encode, axis=1)
                pool.close()
                pool.join()

            all_mv = all_mv.astype('int16')
            train_mv, test_mv = all_mv.iloc[:label.shape[0], :].reset_index(drop=True), all_mv.iloc[label.shape[0]:, :].reset_index(drop=True)
            del all_mv

            train_feature.append(train_mv)
            test_feature.append(test_mv)

            module_end = time.time()
            self.module_time['encode_mv'][self.batch_num] = module_end - module_start

        feature_new = pd.concat(train_feature, axis=1)
        for feat in train_feature:
            del feat
        del train_feature

        self.X_new = pd.concat(test_feature, axis=1)
        for feat in test_feature:
            del feat
        del test_feature

        weight = None

        # time spent on data/feature processing in current batch
        time_spent_invariant, time_spent_variant = 0., 0.
        for m in ['read_data', 'preprocess']:
            time_spent_invariant += self.module_time[m][self.batch_num]
        for m in ['encode_cat', 'encode_mv']:
            time_spent_variant += self.module_time[m][self.batch_num]
        
        # estimated time budget for each remaining batch
        time_left = data_info['time_budget'] - (self.predict_start_time - time_info[1])
        batch_left = self.total_batch_num - self.batch_num
        time_per_batch = time_left / batch_left

        # estimated time budget for training models
        ratio = float(self.batch_window_size + 1) / (min(self.batch_num, self.batch_window_size) + 1)
        time_for_model = time_per_batch - time_spent_invariant - time_spent_variant * ratio

        clf = GBM(category_cols=None, hyper_tune=False, hyper_param=self.hyper_param)

        if (self.batch_num == 1):
            # decide boosting iterations
            self.suggested_boost_round = self.time_manager.suggest_boosting_round(feature_new, label, time_for_model, self.batch_num)
            self.suggested_learning_rate, self.early_stop_round = clf.suggest_learning_rate(feature_new, label, self.suggested_boost_round)
            self.num_boost_round = self.early_stop_round
            print ('max boost round suggested by time manager: %d' %(self.suggested_boost_round))
            print ('early stop round: %d' %(self.early_stop_round))
            print ('suggested learning rate: %f' %(self.suggested_learning_rate))
        else:
            # adjust boosting round number according to time budget
            ratio = min(self.batch_window_size, self.batch_num) / (min(self.batch_window_size, self.batch_num) - 1)
            estimated_train_time = self.module_time['fit'][self.batch_num - 1] * ratio
            remaining_time = time_for_model - estimated_train_time
            if (remaining_time > time_for_model * 0.1 and self.num_boost_round < 2000):
                self.num_boost_round += 200
            elif (remaining_time < 0.):
                self.num_boost_round -= 100

        print ('training info of batch %d: ' %(self.batch_num))
        print ('feature number: %d' %(feature_new.shape[1]))
        print ('train sample size: %d' %(feature_new.shape[0]))
        print ('test sample size: %d' %(self.X_new.shape[0]))
        print ('num boost round: %d' %(self.num_boost_round))
        print ('learning rate: %f' %(self.suggested_learning_rate))

        # LGB
        module_start = time.time()

        updated_params = {'n_estimators': self.num_boost_round, 'learning_rate': self.suggested_learning_rate}
        self.hyper_param = clf.fit(feature_new, label, weight=weight, updated_params=updated_params)
        
        module_end = time.time()
        self.module_time['fit'][self.batch_num] = module_end - module_start

        module_start = time.time()
        
        y_pred = clf.predict(self.X_new)
        
        module_end = time.time()
        self.module_time['predict'][self.batch_num] = module_end - module_start

        self.time_spent_ratio = (time.time() - self.predict_start_time) / time_per_batch * 100.

        self.is_trained = True

        return y_pred

    def predict(self, F, data_info, time_info):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return random values...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. 
        The function predict eventually can return probabilities or continuous values.
        '''
        self.batch_num += 1
        self.predict_start_time = time.time()
        self.module_time['read_data'][self.batch_num] = (self.predict_start_time - self.fit_end_time)

        module_start = time.time()

        F['numerical'] = self.missing_value_preprocess.transform(F['numerical'], 'numerical', input_type='ndarray')
        F['CAT'] = self.missing_value_preprocess.transform(F['CAT'], 'CAT', input_type='dataframe')
        if self.use_mv:
            F['MV'] = self.missing_value_preprocess.transform(F['MV'], 'MV', input_type='dataframe')

        F['numerical'] = data_converter.replace_missing(F['numerical']).astype('float32')
        F['CAT'] = F['CAT'].fillna('-1')

        module_end = time.time()
        self.module_time['preprocess'][self.batch_num] = module_end - module_start

        self.F = F

        self.y_pred = self.transferPredict(0, data_info, time_info)
        
        self.predict_end_time = time.time()
        return self.y_pred

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self