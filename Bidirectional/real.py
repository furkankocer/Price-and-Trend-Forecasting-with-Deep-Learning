import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import LSTM,GRU,Dense,Dropout,Activation
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from keras import backend
from subprocess import check_output
import time
from keras.callbacks import Callback
from keras.models import load_model
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

myArr=[]
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):

        Xt = model.predict(X_test)
        act = []
        pred = []

        
        
        for i in range (len(X_test)):
        
            Xt = model.predict(X_test[i].reshape(1,15,1))
            print('predicted:{0}, actual:{1}'.format(scl.inverse_transform(Xt),scl.inverse_transform(y_test[i].reshape(-1,1))))
            pred.append(scl.inverse_transform(Xt))
            act.append(scl.inverse_transform(y_test[i].reshape(-1,1)))

        a=0
        b=0
        d=0
        c=0
        

        
        #Directional Accuracy CoMatrix
        for i in range(0,299):
            if act[i]>act[i+1] and act[i]<pred[i+1]:
                 b=b+1
                

            elif(act[i]>act[i+1] and act[i]>pred[i+1]):
                a=a+1
                
            elif(act[i]<act[i+1] and act[i]<pred[i+1]):
                d=d+1
            
            else:
                c=c+1

        dacc=(a+d)/(a+b+c+d)
        print("SELL AND BUY",b)
        print("SELL AND SELL",a)
        print("BUY AND BUY",d)  
        print("BUY AND SELL ",c)
        print("Directional Accuracy Per Epochs",dacc)

        myArr.append(dacc)
       

        
        
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
X_train, X_test, y_train, y_test = train_test_split(data, cl,test_size=0.023728545, shuffle=False)


print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_:", y_test.shape)

start = time.time()
model = Sequential()
model.add(Bidirectional(LSTM(64,return_sequences = True,input_shape=(15,1))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(16,return_sequences = True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64,return_sequences = False)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('tanh'))
model.compile(optimizer='adam',loss='mean_squared_error')
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
history=model.fit(X_train, y_train,epochs=100,batch_size=2,validation_data=(X_train, y_train),callbacks=[TestCallback((X_test, y_test))])
model.save('lstm_model.h5')
Xt = model.predict(X_test)
testScore = math.sqrt(mean_squared_error(y_test, Xt))
end = time.time()

total=0

for i in range (0,len(myArr)):
	total=total+myArr[i]
    
average=total/100 
print('Total Directional Accuracy is : ',average)

print('Test Score: %.2f RMSE' % (testScore))
print("completed in ", round(end-start,4),"seconds")

plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
plt.plot(scl.inverse_transform(Xt))
plt.legend(['TRAIN', 'TEST'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.title('LSTM loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(myArr)
plt.title('Directional Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()



