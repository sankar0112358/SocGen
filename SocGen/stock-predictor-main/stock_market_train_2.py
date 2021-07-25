# importing the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout

# l the train dataset
dataset_train = pd.read_csv('sg_train.csv')
dataset_train_copy = dataset_train.copy()
dataset_train_copy = dataset_train_copy.sort_values(by='Date')

# removing the columns other than open, close, high and low
dataset_train_copy = dataset_train_copy[['Date','Open','High','Low','Close']]
dataset_train_copy = dataset_train_copy.dropna(axis=0)
train_data = dataset_train_copy[['Open','High','Low','Close']]

# Feature Scaling
sc = MinMaxScaler(feature_range=(0,1))
train_data = sc.fit_transform(train_data)

# Creating a data structure with 60 time steps and 1 output
X_train = []
y_train = []
for i in range(60,4950):
    X_train.append(train_data[i-60:i,:])
    y_train.append(train_data[i,:])
X_train, y_train = np.array(X_train), np.array(y_train)

# Building the RNN
regressor = Sequential()

regressor.add(GRU(units=118,activation='relu',return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
regressor.add(Dropout(0.2))

regressor.add(GRU(units=118,activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(GRU(units=118,activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(GRU(units=118,activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(GRU(units=118,activation='relu'))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=X_train.shape[2]))

regressor.compile(optimizer='adam',loss='mean_squared_error')

# printing the summary of RNN
print(regressor.summary())

# fitting the RNN to train data
regressor.fit(X_train,y_train,epochs=100, batch_size=50)

# saving the model
model_json = regressor.to_json()
with open('regressor.json', 'w') as json_file:
    json_file.write(model_json)

regressor.save_weights("model.h5")
print("Saved model to disk")






