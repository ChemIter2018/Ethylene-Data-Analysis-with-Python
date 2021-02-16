import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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
# If value < 1, SVR Can't calculate normally, So multiply the value by 1000.
FurnaceDataPE = FurnaceData.iloc[:, 26] * 1000

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
sns.set_theme(style='darkgrid', font='Arial', font_scale=1.5)
plt.subplots(figsize=(16, 13))

g = sns.scatterplot(x=y_test / 1000, y=y_test_predict / 1000)
g.set(xlim=(0.48, 0.55))
g.set(ylim=(0.48, 0.55))

sns.histplot(x=y_test / 1000, y=y_test_predict / 1000, bins=50, pthresh=.1, cmap="mako",
             cbar=True, cbar_kws=dict(shrink=.75))
sns.kdeplot(x=y_test / 1000, y=y_test_predict / 1000, levels=5, color="w", linewidths=1.5)

sns.despine(trim=True, left=True)

qq = np.linspace(np.min(((y_test / 1000).min(), (y_test_predict / 1000).min())),
                 np.max(((y_test / 1000).max(), (y_test_predict / 1000).max())))
plt.plot(qq, qq, color="navy", ls="--", linewidth=2)

plt.xlabel('Measured Value', fontsize=30, fontweight='bold')
plt.ylabel('Predicted Value', fontsize=30, fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
plt.savefig('01SVR_PE.png', dpi=600)

# Run Time
finish_time = time()
run_time = finish_time - start_time
print("Total program running time: ", run_time)
