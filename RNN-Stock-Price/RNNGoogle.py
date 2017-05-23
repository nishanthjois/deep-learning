
"""
Created on Tue May 23 19:59:15 2017

@author: nj
"""

import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# 1. Data preprocessing
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values #add :2 to create into matix

# Feature scaling
# In LSTM we use sigmoid function hence we will use normalization (which gives +1 to -1)
# instead of standardization 

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
training_set=sc.fit_transform(training_set)

# Get inputs and outputs
X_train = training_set[0:1257] # first to last but one
y_train=training_set[1:1258]


#Reshaping
# Input is 2-d array 1257 observations and 1 feature
# Reshape is to change format to 3d array - we will add time step which is 1 in this case
# Keras documentation: 3D tensor with shape (batch_size, timestep, input_dim); batch size corresponds to
# to number of observations, timestep is 1 day here, and input_dim is number of features

X_train = np.reshape(X_train,(1257,1,1))


# 2. Building RNN - Regression model

regressor = Sequential()

# units = number of memory units, 
# input_shape = (None,1) :: 1 corresponds to number of features
# None corresponds time steps (it can be none which means any number of )
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
#regressor.add(LSTM(32, input_shape=(None, 1)))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(X_train,y_train,batch_size=32,epochs=200)

# 3. Making predictions

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

inputs = real_stock_price
inputs=sc.transform(inputs)
inputs = np.reshape(inputs,(20,1,1))
predicted_stock_price=regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)





















