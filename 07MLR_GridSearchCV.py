# -*- coding: utf-8 -*-
# @Time    : 2021/2/20 21:14
# @Author  : Chemiter
# @FileName: 07MLR_GridSearchCV.py
# @Software: PyCharm

import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from time import *

start_time = time()
# Read Furnace Data
FurnaceData = pd.read_csv("00FurnaceCleanData.csv")
FurnaceDataX = FurnaceData.iloc[:, 1:20]
FurnaceDataX_Scale = preprocessing.scale(FurnaceDataX)
FurnaceDataPE = FurnaceData.iloc[:, 26]

# Split Train and Test Data
# random_state: RandomState instance or None, default=None
X_train, X_test, y_train, y_test = train_test_split(FurnaceDataX_Scale, FurnaceDataPE, random_state=0, test_size=0.3)

parameters = [{'activation': ('identity', 'logistic', 'tanh', 'relu'),
               'solver': ('lbfgs', 'sgd', 'adam'),
               'learning_rate':('constant', 'invscaling', 'adaptive'),
               'alpha':[0.0001, 0.001, 0.01, 0.1]}]

mlp = MLPRegressor()
clf = GridSearchCV(mlp, parameters, cv=5)
clt_result = clf.fit(X_train, y_train)

best_params = clt_result.best_params_

print(best_params)

# Run Time
finish_time = time()
run_time = finish_time - start_time
print("Total program running time: ", run_time)
