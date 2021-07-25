# importing the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json

# loading the train data
dataset_train = pd.read_csv('sg_train.csv')
dataset_train_copy = dataset_train.copy()
dataset_train_copy = dataset_train_copy.sort_values(by='Date')

# removing the columns other than open, close, high and low
dataset_train_copy = dataset_train_copy[['Date','Open','High','Low','Close']]
dataset_train_copy = dataset_train_copy.dropna(axis=0)
train_data = dataset_train_copy[['Open','High','Low','Close']]

# Feature Scaling
sc = MinMaxScaler(feature_range=(0,1))
sc.fit(train_data)

# opening the model files
json_file = open('regressor.json', 'r')
loaded_model_json_open = json_file.read()
json_file.close()
regressor = model_from_json(loaded_model_json_open)
# load weights into new model
regressor.load_weights("model.h5")
print("Loaded model from disk")

# printing the summary
print(regressor.summary())

# making predictions and visualizing the results
dataset_test = pd.read_csv('sg_test.csv')
dataset_test = dataset_test.sort_values(by='Date')
real_stock_price = dataset_test.iloc[:,0:5].values

#  getting predicted stock price of July 2021
dataset_total = pd.concat([dataset_train_copy,dataset_test.iloc[:,0:5]],axis=0)
inputs = dataset_total.iloc[len(dataset_total)-len(dataset_test)-60:,1:5].values
inputs = sc.transform(inputs)
X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,:])
X_test = np.array(X_test)
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# saving the predictions
dataset_test_predict = pd.concat([dataset_test.iloc[:,0],pd.DataFrame(data=predicted_stock_price,columns=['Open','High','Low','Close'])],axis=1)
dataset_test_predict.to_csv('sg_test_predict.csv')




