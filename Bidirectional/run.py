import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from keras import backend
from subprocess import check_output
import time
from keras.callbacks import Callback
from keras.models import load_model

dataset = pd.read_csv('5data.csv')
# index is dropped
dataset = dataset.drop(dataset.index[0])
# date axis is dropped using drop function
dataset = dataset.drop(['date'], axis=1)
# iloc is used for index where loc is used for label
data = dataset.iloc[:, 1:]
cl = dataset.iloc[:, 0]

# convert dataframe in numpy array
data = data.values
data = data.astype('float64')
scl = MinMaxScaler()
#Scale the data
cl = cl.values.reshape(cl.shape[0],1)
cl = scl.fit_transform(cl)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, cl,test_size=0.02372854544, shuffle=False)

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
model = load_model('lstm_model.h5')

Xt = model.predict(X_test)

plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
plt.plot(scl.inverse_transform(Xt))
plt.legend(['TRAIN', 'TEST'], loc='upper left')
plt.show()

act = []
pred = []
for i in range (len(X_test)):
        
            Xt = model.predict(X_test[i].reshape(1,15,1))
            print('predicted:{0}, actual:{1}'.format(scl.inverse_transform(Xt),scl.inverse_transform(y_test[i].reshape(-1,1))))
            pred.append(scl.inverse_transform(Xt))
            act.append(scl.inverse_transform(y_test[i].reshape(-1,1)))



