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

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

class GBM:

    def __init__(self, category_cols=None, hyper_tune=False, hyper_param=None):

        self.clf = None
        self.category_cols = category_cols
        self.hyper_tune = hyper_tune
        self.hyper_param = hyper_param
        
        self.default_hyper_param = {
            'num_leaves': 31, 
            'n_estimators': 1000, 
            'learning_rate': 0.01, 
            'colsample_bytree': 1.0, 
            #'subsample': 0.05, 
            'min_child_weight': 5, 
            'scale_pos_weight': 1.0, 
            'reg_alpha': 5.0, 
            'reg_lambda': 5.0
        }

    def setParams(self, config):
        params = {
            'task': 'train',
            'boosting_type': 'goss', 
            'objective': 'binary', 
            'metric': 'auc', 
            'num_leaves': int(config["num_leaves"]), 
            'learning_rate': config["learning_rate"], 
            'feature_fraction': config["colsample_bytree"], 
            #'bagging_fraction': self.hyper_param["subsample"], 
            #'bagging_freq': 1, 
            'min_data_in_leaf': int(config["min_child_weight"]), 
            'lambda_l1': config["reg_alpha"], 
            'lambda_l2': config["reg_lambda"], 
            'top_rate': 0.1, 
            'other_rate': 0.05, 
            #'is_unbalance': True, 
            'scale_pos_weight': config['scale_pos_weight'], 
            #'num_threads': 20, 
            'verbose': -1
        }
        return params


    def evaluateParams(self, config):
        params = self.setParams(config)
        gbm = lgb.train(params, self.train_set, num_boost_round=int(config['n_estimators']))
        metric = roc_auc_score(self.y_valid, gbm.predict(self.X_valid))
        print ('auc: ', 2 * metric - 1)
        return (1 - metric)


    def hyperTune(self, X, y):
        
        #X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=0)
        skf = StratifiedKFold(n_splits=2, shuffle=True)
        for train_idx, valid_idx in skf.split(X, y):
            self.X_train, self.X_valid = X[train_idx], X[valid_idx]
            self.y_train, self.y_valid = y[train_idx], y[valid_idx]
        self.train_set = lgb.Dataset(self.X_train, self.y_train, free_raw_data=False).construct()
        
        # hyperopt
        space = {'num_leaves':hp.quniform('num_leaves', 8, 64, 2), 
                 "n_estimators":hp.quniform("n_estimators", 50, 2000, 50), 
                 'learning_rate':hp.loguniform('learning_rate', np.log(0.001), np.log(0.5)), 
                 #"subsample":hp.uniform("subsample", 0.5, 1.0), 
                 "colsample_bytree":hp.uniform("colsample_bytree", 0.5, 1.0), 
                 "min_child_weight":hp.quniform('min_child_weight', 0, 500, 10), 
                 'scale_pos_weight': hp.uniform('scale_pos_weight', 1.0, 10.0), 
                 "reg_alpha":hp.uniform("reg_alpha", 0.0, 2.0), 
                 "reg_lambda":hp.uniform("reg_lambda", 0.0, 2.0)
                }
        algo = partial(tpe.suggest, n_startup_jobs=5)
        self.hyper_param = fmin(self.evaluateParams, space, algo=algo, max_evals=50)
        print ('ho params: ', self.hyper_param)
        return self.hyper_param

    def fit(self, X, y, weight=None, updated_params=None):

        if (self.hyper_param == None):
            if (self.hyper_tune == False):
                self.hyper_param = self.default_hyper_param
            else:
                self.hyper_param = self.hyperTune(X, y)

        if (updated_params != None):
            self.hyper_param.update(updated_params)

        params = self.setParams(self.hyper_param)

        data = lgb.Dataset(X, y, weight=weight)
        self.clf = lgb.train(params, data, num_boost_round=int(self.hyper_param['n_estimators']))
        return self.hyper_param

    def predict(self, X):
        return self.clf.predict(X)

    def suggest_learning_rate(self, X, y, max_boost_round):
        
        lr = [0.01, 0.02, 0.03, 0.04, 0.05]
        
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

        params = self.setParams(self.default_hyper_param)

        max_round = max_boost_round // 500
        auc = np.zeros([len(lr), max_round])
        for i in range(len(lr)):
            print ('learning rate: %.2f' %(lr[i]))
            params['learning_rate'] = lr[i]
            train_data = lgb.Dataset(X_train, y_train, free_raw_data=False)
            clf = None
            for j in range(max_round):
                clf = lgb.train(params, train_data, num_boost_round=500, init_model=clf, keep_training_booster=True)
                # score with regularization
                auc[i, j] = roc_auc_score(y_valid, clf.predict(X_valid)) - lr[i] * 0.1 + j * 0.001

        print (auc)
        idx = np.argmax(auc)
        best_lr = lr[idx // max_round]
        best_boost_round = (idx % max_round + 1) * 500
        return best_lr, best_boost_round