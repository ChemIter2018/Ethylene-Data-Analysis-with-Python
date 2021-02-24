# -*- coding: utf-8 -*-
# @Time    : 2021/2/18 23:55
# @Author  : Chemiter
# @FileName: 03KNR_GridSearchCV.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import json
import joblib

from sklearn.metrics import r2_score
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from time import *

start_time = time()
# Read Furnace Data
FurnaceData = pd.read_csv("00FurnaceCleanData.csv")
FurnaceDataX = FurnaceData.iloc[:, 1:20]

# If value < 1, KNR Can't calculate normally, So multiply the value by 100.
n = 100
FurnaceDataPE = FurnaceData.iloc[:, 26] * n

# Split Train and Test Data
# random_state: RandomState instance or None, default=None
X_train, X_test, y_train, y_test = train_test_split(FurnaceDataX, FurnaceDataPE, random_state=0, test_size=0.3)

# StandardScaler Data
ScalerSave = preprocessing.StandardScaler().fit(FurnaceDataX)
# ScalerSave = pickle.load(open('Saved_Model/Keras_MLP_ScalerSave.pkl', 'rb'))

X_train = ScalerSave.transform(X_train)
X_test = ScalerSave.transform(X_test)
pickle.dump(ScalerSave, open('Saved_Model/SL_KNR_ScalerSave.pkl', 'wb'))

parameters = [{'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'weights': ['uniform', 'distance']}]

knr = neighbors.KNeighborsRegressor()
clf = GridSearchCV(knr, parameters, cv=5)
clt_result = clf.fit(X_train, y_train)

best_params = clt_result.best_params_
f = open("Saved_Model/SL_KNR_best_params.txt", "wb")
f.write(json.dumps(best_params).encode())
f.close()

# KNR
knr_reg = neighbors.KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
knr_reg.fit(X_train, y_train)

joblib.dump(knr_reg, 'Saved_Model/scikit-learn_KNR.model')
# knr_reg = joblib.load('Saved_Model/scikit-learn_KNR.model')

# Test Data Predict
y_test_predict = knr_reg.predict(X_test)
R2 = r2_score(y_test, y_test_predict)

# Plot Q-Q
sns.set_theme(style='darkgrid', font='Arial', font_scale=1.5)
plt.subplots(figsize=(16, 13))

y_test = np.array(y_test / n)
y_test_predict = y_test_predict.flatten() / n

g = sns.scatterplot(x=y_test, y=y_test_predict)
g.set(xlim=(0.48, 0.55))
g.set(ylim=(0.48, 0.55))

h = sns.histplot(x=y_test, y=y_test_predict, bins=50, pthresh=.1, cmap="mako",
                 cbar=True, cbar_kws=dict(shrink=1))

sns.kdeplot(x=y_test, y=y_test_predict, levels=5, color="w", linewidths=1.5)

sns.despine(trim=True, left=True)

qq = np.linspace(np.min((y_test.min(), y_test_predict.min())),
                 np.max((y_test.max(), y_test_predict.max())))
plt.plot(qq, qq, color="navy", ls="--", linewidth=2.5)

# Run Time
finish_time = time()
run_time = finish_time - start_time

plt.text(0.482, 0.5405, "$\mathregular{R^2}$" + " = "
         + str(round(R2, 4)), fontsize=25, fontweight='bold')
plt.text(0.482, 0.544, "Training time: "
         + str(round(run_time, 4)) + "s", fontsize=25, fontweight='bold')
plt.text(0.482, 0.547, "Modeling method: " + "KNR(scikit-learn)", fontsize=25, fontweight='bold')

plt.xlabel('Measured Value', fontsize=30, fontweight='bold')
plt.ylabel('Predicted Value', fontsize=30, fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
plt.savefig('Pictures/04SL_KNR_PE.png', dpi=1000)
