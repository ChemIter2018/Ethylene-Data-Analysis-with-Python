import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from time import *

start_time = time()
# Read Furnace Data
FurnaceData = pd.read_csv("00FurnaceCleanData.csv")
FurnaceDataX = FurnaceData.iloc[:, 1:20]
FurnaceDataX_Scale = preprocessing.scale(FurnaceDataX)
FurnaceDataPE = FurnaceData.iloc[:, 26]*100

# Split Train and Test Data
# random_state: RandomState instance or None, default=None
X_train, X_test, y_train, y_test = train_test_split(FurnaceDataX_Scale, FurnaceDataPE, random_state=0, test_size=0.3)

# SVR
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
svm_poly_reg.fit(X_train, y_train)

# Test Data Predict
y_test_predict = svm_poly_reg.predict(X_test)

# Measure Regression Performance
print(r2_score(y_test, y_test_predict))

# Plot
plt.figure(figsize=(9, 9))
plt.plot(y_test/100, y_test_predict/100, '.')
plt.xlabel('Measured Value')
plt.ylabel('Predicted Value')
plt.title("P/E", fontsize=20)
plt.savefig('01SVR_PE.png', dpi=600)

# Run Time
finish_time = time()
run_time = finish_time - start_time
print("Total program running time: ", run_time)

