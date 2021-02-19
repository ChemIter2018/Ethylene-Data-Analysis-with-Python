# -*- coding: utf-8 -*-
# @Time    : 2021/2/18 23:55
# @Author  : Chemiter
# @FileName: 03KNR_GridSearchCV.py
# @Software: PyCharm

import pandas as pd

from sklearn import neighbors
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

parameters = [{'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'weights': ['uniform', 'distance']}]

knr = neighbors.KNeighborsRegressor()
clf = GridSearchCV(knr, parameters, cv=5)
clt_result = clf.fit(X_train, y_train)

best_params = clt_result.best_params_

print(best_params)

# Run Time
finish_time = time()
run_time = finish_time - start_time
print("Total program running time: ", run_time)

