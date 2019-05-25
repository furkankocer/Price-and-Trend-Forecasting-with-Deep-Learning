# Price-and-Trend-Forecasting-with-Deep-Learning
One of the most critical application areas in the Financial Market especially sits on Stock Markets. 
In this area, the aim is trying to predict the future value of a specific stock by looking at its previous financial data on the exchange process in the market. 
We proposed a system that uses a Deep Learning based approach for training and constructing a knowledge base on a specific stock such as "IBM".  
We get time series values of the stock from the New York Stock Exchange which starts from 1968 up to 2018. 
Experimental results showed that this approach produces very good forecasting for specific stocks.

Proposed System of time series data, tanh activation function was used for hidden BLSTM layers during model training. Different training algorithms were tested and finally, the Adam training algorithm was selected and as its learning speed 0.001 and its batch size 2. Network inputs are presented as 15 data points vectors corresponding to past values of the last five days.
The number of hidden layers and the number of neurons corresponding set experimentally. 64,16 and 64 which corresponds to the number of neurons BLSTM three hidden layer showed the lowest error estimate and the highest directional accuracy with the comparison of other BLSTM models. On the other hand, 0.2 dropouts is used to prevent overfitting for each layer.

After the compilation of the model, for testing the 300 days prediction 
We can easily compute that the directional accuracy of this model is 0,64 at 100th epochs. 
On each epoch accuracy has been fluctuated because of the dropout layer.
 
