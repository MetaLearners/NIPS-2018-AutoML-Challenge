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

import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, QuantileTransformer
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def seperate(x):
    try:
        x = tuple(x.split(','))
    except AttributeError:
        x = ('-1', )
    return x


class MVEncoder:

    def __init__(self, max_cat_num=1000):
        self.max_cat_num = max_cat_num

    def encode(self, cats):
        return min((self.mapping[c] for c in cats))

    def fit_transform(self, X):
        
        X = X.map(seperate)

        cat_count = {}
        for cats in X:
            for c in cats:
                try:
                    cat_count[c] += 1
                except KeyError:
                    cat_count[c] = 1
        cat_list = np.array(list(cat_count.keys()))
        cat_num = np.array(list(cat_count.values()))
        idx = np.argsort(-cat_num)
        cat_list = cat_list[idx]

        self.mapping = {}
        for i, cat in enumerate(cat_list):
            self.mapping[cat] = min(i, self.max_cat_num)
        del cat_count, cat_list, cat_num

        X_encode = X.map(self.encode)

        return X_encode


class CATEncoder:

    def __init__(self):
        pass

    def fit_transform(self, X):
        col = X.columns[0]
        count = X.groupby(col).size().reset_index().rename(columns={0: col + '_count'})
        X = X.merge(count, how='left', on=col)
        X_encode = X[col + '_count'].astype('int32')
        del X, count
        return X_encode


class MissingValuePreprocess:

    def __init__(self, drop_ratio=0.6):
        self.drop_ratio = drop_ratio
        self.use_columns = {}

    def fit_transform(self, X, feat_type, input_type='dataframe'):
        if (input_type == 'dataframe'):
            missing_ratio = X.isnull().sum(axis=0).values.astype('float') / X.shape[0]
        else:
            missing_ratio = np.isnan(X).sum(axis=0).astype('float') / X.shape[0]
        print (missing_ratio)
        self.use_columns[feat_type] = np.arange(X.shape[1])[missing_ratio < self.drop_ratio]
        return self.transform(X, feat_type, input_type=input_type)

    def transform(self, X, feat_type, input_type='dataframe'):
        if (input_type == 'dataframe'):
            return X.iloc[:, self.use_columns[feat_type]]
        else:
            return X[:, self.use_columns[feat_type]]

    def count_missing_value(self, X, input_type):
        count = np.zeros(X[0].shape[0], dtype='int')
        for i, feat in enumerate(X):
            if (input_type[i] == 'dataframe'):
                count += X[i].isnull().sum(axis=1).values
            else:
                count += np.isnan(X[i]).sum(axis=1)
        return count.reshape(-1, 1)