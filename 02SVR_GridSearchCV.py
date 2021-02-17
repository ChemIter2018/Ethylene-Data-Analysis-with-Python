import pandas as pd

from sklearn.svm import SVR
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
n = 1000
FurnaceDataPE = FurnaceData.iloc[:, 26] * n

# Split Train and Test Data
# random_state: RandomState instance or None, default=None
X_train, X_test, y_train, y_test = train_test_split(FurnaceDataX_Scale, FurnaceDataPE, random_state=0, test_size=0.3)

parameters = {'kernel': ('poly', 'rbf'), 'C': [10, 100, 1000],
              'gamma': ('auto', 'scale'), 'degree': [3, 5, 7],
              'epsilon': [0.05, 0.1, 0.5]}

svr = SVR()
clf = GridSearchCV(svr, parameters, cv=5)
clt_result = clf.fit(X_train, y_train)

best_params = clt_result.best_params_

print(best_params)

# Run Time
finish_time = time()
run_time = finish_time - start_time
print("Total program running time: ", run_time)