# -*- coding: utf-8 -*-
# @Time    : 2021/2/19 22:25
# @Author  : Chemiter
# @FileName: 05DTR_gridSearchCV.py
# @Software: PyCharm

import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from time import *

start_time = time()
# Read Furnace Data
FurnaceData = pd.read_csv("00FurnaceCleanData_SVR_GridSearchCV.csv")
FurnaceDataX = FurnaceData.iloc[:, 1:20]
FurnaceDataX_Scale = preprocessing.scale(FurnaceDataX)

# If value < 1, SVR Can't calculate normally, So multiply the value by 1000.
n = 100
FurnaceDataPE = FurnaceData.iloc[:, 26] * n

# Split Train and Test Data
# random_state: RandomState instance or None, default=None
X_train, X_test, y_train, y_test = train_test_split(FurnaceDataX_Scale, FurnaceDataPE, random_state=0, test_size=0.3)

# parameters = {'criterion': ('mse', 'friedman_mse', 'mae'), 'splitter': ('best', 'random'),
#               'max_depth': [100, 1000, 10000, 15000, 20000]}
parameters = {'n_estimators': [100, 300, 500, 700, 900]}

dtr = AdaBoostRegressor(DecisionTreeRegressor())
clf = GridSearchCV(dtr, parameters, cv=5)
clt_result = clf.fit(X_train, y_train)

best_params = clt_result.best_params_

print(best_params)

# Run Time
finish_time = time()
run_time = finish_time - start_time
print("Total program running time: ", run_time)