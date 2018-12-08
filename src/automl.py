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
import time


class TimeManager:

    def __init__(self, budget_score, max_batch_num):
        self.budget_score = budget_score
        self.max_batch_num = max_batch_num
        self.base_round_num = 500
        self.max_round = 4

    def suggest_boosting_round(self, X, y, time_bedget, batch_num):
        
        if (self.budget_score < 3.):
            self.max_round = 2

        ratio = float(self.max_batch_num) / batch_num

        params = {
            'task': 'train',
            'boosting_type': 'goss', 
            'objective': 'binary', 
            'metric': 'auc', 
            'num_leaves': 31, 
            'learning_rate': 0.01, 
            'feature_fraction': 1.0, 
            'min_data_in_leaf': 5, 
            'top_rate': 0.1, 
            'other_rate': 0.05, 
            #'num_threads': 20, 
            'verbose': -1
        }
        data = lgb.Dataset(X, y)
        
        train_start = time.time()
        clf = lgb.train(params, data, num_boost_round=self.base_round_num)
        train_end = time.time()
        
        estimated_train_time = (np.arange(self.max_round) + 1) * (train_end - train_start) * ratio
        idx = np.arange(self.max_round)[estimated_train_time <= time_bedget]

        if (idx.shape[0] == 0):
            self.suggested_boost_round = self.base_round_num
            self.suggested_train_time = estimated_train_time[0]
        else:
            self.suggested_boost_round = (idx[-1] + 1) * self.base_round_num
            self.suggested_train_time = estimated_train_time[idx[-1]]

        return self.suggested_boost_round