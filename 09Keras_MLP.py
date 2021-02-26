# -*- coding: utf-8 -*-
# @Time    : 2021/2/21 16:32
# @Author  : Chemiter
# @FileName: 09Keras_MLP.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pickle

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from time import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

start_time = time()
# Read Furnace Data
FurnaceData = pd.read_csv("00FurnaceCleanData.csv")
FurnaceDataX = FurnaceData.iloc[:, 1:20]

# If value < 1, MLP Can't calculate normally, So multiply the value by 100.
n = 100
FurnaceDataPE = FurnaceData.iloc[:, 26] * n

# Split Train and Test Data
# random_state: RandomState instance or None, default=None
X_train, X_test, y_train, y_test = train_test_split(FurnaceDataX, FurnaceDataPE, random_state=1, test_size=0.3)

# StandardScaler Data
ScalerSave = preprocessing.StandardScaler().fit(FurnaceDataX)
# ScalerSave = pickle.load(open('Saved_Model/Keras_MLP_ScalerSave.pkl', 'rb'))
X_train = ScalerSave.transform(X_train)
X_test = ScalerSave.transform(X_test)
pickle.dump(ScalerSave, open('Saved_Model/Keras_MLP_ScalerSave.pkl', 'wb'))

# Training Model
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:], name="layer1"),
    keras.layers.Dense(15, activation="relu", name="layer2"),
    keras.layers.Dense(1, name="layer3")
])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=1e-3), metrics=["mean_squared_error"])
model.fit(X_train, y_train, epochs=30000, batch_size=10000)

# Save or Load Model
model.save('Saved_Model/Keras_MLP')
# model = tf.keras.models.load_model('Saved_Model/Keras_MLP')

y_test_predict = model.predict(X_test)
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
plt.text(0.482, 0.547, "Modeling method: " + "MLP(Keras)", fontsize=25, fontweight='bold')

plt.xlabel('Measured Value', fontsize=30, fontweight='bold')
plt.ylabel('Predicted Value', fontsize=30, fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
plt.savefig('Pictures/09TF_MLP_PE.png', dpi=1000)
