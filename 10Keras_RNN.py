# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 23:37
# @Author  : Chemiter
# @FileName: 10Keras_RNN.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
import pickle

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from time import *

start_time = time()
# Read Furnace Data
FurnaceData = pd.read_csv("00FurnaceCleanData.csv")
FurnaceDataX = FurnaceData.iloc[:, 1:20]
ScalerSave = preprocessing.StandardScaler().fit(FurnaceDataX)
FurnaceDataX_Scale = ScalerSave.transform(FurnaceDataX)
pickle.dump(ScalerSave, open('Saved_Model/Keras_RNN_ScalerSave.pkl', 'wb'))
# ScalerSave = pickle.load(open('Saved_Model/Keras_RNN_ScalerSave.pkl', 'rb'))
# X_test = ScalerSave.transform(X_test)

# If value < 1, RNN Can't calculate normally, So multiply the value by 100.
n = 100
FurnaceDataPE = FurnaceData.iloc[:, 26] * n

# Split Train and Test Data
# random_state: RandomState instance or None, default=None
X_train, X_test, y_train, y_test = train_test_split(FurnaceDataX_Scale, FurnaceDataPE, random_state=0, test_size=0.3)
X_train = np.reshape(X_train, (len(X_train), 1, 19))
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.LSTM(30, activation="tanh", input_shape=[None, 19], return_sequences=True, name="layer1"),
    keras.layers.LSTM(15, return_sequences=True, name="layer2"),
    keras.layers.LSTM(7, return_sequences=True, name="layer3"),
    keras.layers.LSTM(3, name="layer4"),
    keras.layers.Dense(1, name="layer5")
])
model.summary()
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=1e-3), metrics=["mean_squared_error"])
history = model.fit(X_train, y_train, epochs=2000)

# Save Model
model.save('Saved_Model/Keras_RNN')
# model = tf.keras.models.load_model('Saved_Model/Keras_RNN')

X_test = np.reshape(X_test, (len(X_test), 1, 19))
y_test_predict = model.predict(X_test)

print(r2_score(y_test, y_test_predict))

# Plot
sns.set_theme(style='darkgrid', font='Arial', font_scale=1.5)
plt.subplots(figsize=(16, 13))

y_test = np.array(y_test/n)
y_test_predict = y_test_predict.flatten()/n

g = sns.scatterplot(x=y_test, y=y_test_predict)
g.set(xlim=(0.48, 0.55))
g.set(ylim=(0.48, 0.55))

sns.histplot(x=y_test, y=y_test_predict, bins=50, pthresh=.1, cmap="mako",
             cbar=True, cbar_kws=dict(shrink=.75))
sns.kdeplot(x=y_test, y=y_test_predict, levels=5, color="w", linewidths=1.5)

sns.despine(trim=True, left=True)

qq = np.linspace(np.min((y_test.min(), y_test_predict.min())),
                 np.max((y_test.max(), y_test_predict.max())))
plt.plot(qq, qq, color="navy", ls="--", linewidth=2)

plt.text(0.483, 0.545, "$\mathregular{R^2}$" + " = "
         + str(r2_score(y_test, y_test_predict)), fontsize=20, fontweight='bold')
plt.xlabel('Measured Value', fontsize=25, fontweight='bold')
plt.ylabel('Predicted Value', fontsize=25, fontweight='bold')
plt.xticks(fontsize=25, fontweight='bold')
plt.yticks(fontsize=25, fontweight='bold')
plt.savefig('Pictures/10TF_RNN_PE.png', dpi=600)

# Run Time
finish_time = time()
run_time = finish_time - start_time
print("Total program running time: ", run_time)
