#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import xgboost as xgd
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
plt.style.use('fivethirtyeight')


# In[41]:


pjme  = pd.read_csv(r"D:\Data science\Data\PJME_hourly.csv", index_col=[0], parse_dates=[0])
color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = pjme.plot(style='.', figsize=(15,5), color=color_pal[0], title='PJM East')

pjme.head(3)


# In[42]:


pjme.shape


# In[43]:


pjme


# In[44]:


pjme.head().style.set_properties(subset=['PJME_MW'], **{'background-color': 'dodgerblue'})


# In[45]:


pjme.describe()


# In[46]:


# Let's look at the years in the data set
months = pjme.index.month.unique()
months


# In[48]:


pjme_monthly_avg=pjme['PJME_MW'].resample('m').mean()
pjme_monthly_avg.to_frame()


# In[49]:


pjme_monthly_avg.shape


# # Normalization Process

# In[51]:


from sklearn.preprocessing import MinMaxScaler
def normalize_data(pjme):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(pjme['PJME_MW'].values.reshape(-1,1))
    pjme['PJME_MW'] = normalized_data
    return pjme, scaler

pjme_norm, scaler = normalize_data(pjme)
pjme_norm.shape


# In[52]:


pjme_norm


# In[53]:


pjme.plot(figsize=(16,5),legend=True)
plt.axhspan(0, 1, facecolor='gray', alpha=0.3)

plt.title("Dominion Virginia Power (DOM) - Megawatt Energy Consumption")

plt.show()


# In[54]:


split_date = '2018-03-13'

DOM_train = pjme_norm.loc[pjme_norm.index <= split_date].copy()
DOM_test = pjme_norm.loc[pjme_norm.index > split_date].copy()


# In[55]:


fig, ax = plt.subplots(figsize=(15, 5))
DOM_train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
DOM_test.plot(ax=ax, label='Test Set')
ax.axvline('2018-03-13', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.axhspan(0, 1, facecolor='gray', alpha=0.3)
plt.show()


# # Prepare Data for Training the RNN & LSTM

# In[56]:


def load_data(data, seq_len):
    X_train = []
    y_train = []
    
    for i in range(seq_len, len(data)):
        X_train.append(data.iloc[i-seq_len : i, 0])
        y_train.append(data.iloc[i, 0])
    
    # last 6189 days are going to be used in test
    X_test = X_train[110000:]             
    y_test = y_train[110000:]
    
    # first 110000 days are going to be used in training
    X_train = X_train[:110000]           
    y_train = y_train[:110000]
    # convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # reshape data to input into RNN&LSTM models
    X_train = np.reshape(X_train, (110000, seq_len, 1))
    
    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))
    
    return [X_train, y_train, X_test, y_test]
    


# In[57]:


seq_len = 20 

# Let's create train, test data
X_train, y_train, X_test, y_test = load_data(pjme, seq_len)

print('X_train.shape = ',X_train.shape)
print('y_train.shape = ', y_train.shape)
print('X_test.shape = ', X_test.shape)
print('y_test.shape = ',y_test.shape)


# # Build a RNN model 

# In[58]:


import tensorflow as tf
from keras.layers import Dense,Dropout,SimpleRNN,LSTM
from keras.models import Sequential
rnn_model = Sequential()

rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
rnn_model.add(Dropout(0.15))

rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True))
rnn_model.add(Dropout(0.15))

rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=False))
rnn_model.add(Dropout(0.15))

rnn_model.add(Dense(1))

rnn_model.summary()


# In[59]:


rnn_model.compile(optimizer="adam",loss="MSE")
rnn_model.fit(X_train, y_train, epochs=10, batch_size=1000)


# In[60]:


from sklearn.metrics import r2_score
rnn_predictions = rnn_model.predict(X_test)

rnn_score = r2_score(y_test,rnn_predictions)
print("R2 Score of RNN model = ",rnn_score)


# In[61]:


y_test_inverse = scaler.inverse_transform(y_test.reshape(-1,1))
rnn_predection_inverse=scaler.inverse_transform(rnn_predictions)
# GEt values after inverse transformation
y_test_inverse=y_test_inverse.flatten()
rnn_predections_inverse=rnn_predection_inverse.flatten() 


# In[62]:


last_35346_index_dates=pjme.index[-35346:]
results_RNN=pd.DataFrame({"Date":last_35346_index_dates,'Actual': y_test_inverse, 'Predicted': rnn_predections_inverse})
results_RNN


# In[63]:


plt.figure(figsize=(16,6))
plt.plot(y_test,color='blue',label=' Actual Distribution of PJME Load')
plt.plot(rnn_predictions, alpha=0.7, color='yellow',label='Distribution of PJME Load')
plt.axhspan(0,1, facecolor='gray',alpha=0.3)
plt.title("Predicction made by simple RNN model")
plt.xlabel("Time")
plt.ylabel('Normalized Distribution of PJME Load')
plt.legend()
plt.show()


# # Build as LSTM model

# In[64]:


lstm_model=Sequential()
lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
lstm_model.add(Dropout(0.15))
lstm_model.add(LSTM(40,activation='tanh',return_sequences=True))
lstm_model.add(Dropout(0.15))
lstm_model.add(LSTM(40,activation='tanh',return_sequences=False))
lstm_model.add(Dropout(0.15))
lstm_model.add(Dense(1))
lstm_model.summary()


# In[65]:


lstm_model.compile(optimizer="adam",loss="MSE")
lstm_model.fit(X_train, y_train, epochs=10, batch_size=1000)


# In[66]:


lstm_predictions= lstm_model.predict(X_test)
lstm_score=r2_score(y_test,lstm_predictions)
print("R^2 Score of LSTM model = ",lstm_score)


# In[67]:


y_test_inverse=scaler.inverse_transform(y_test.reshape(-1,1))
lstm_predictins_inverse=scaler.inverse_transform(lstm_predictions)
# Get values after inverse transformation
y_test_inverse = y_test_inverse.flatten()
lstm_predictions_inverse=lstm_predictins_inverse.flatten()


# In[68]:


#Now let's see our actual y and predicted y values as data frames
results_LSTM = pd.DataFrame({"Date": last_35346_index_dates,'Actual': y_test_inverse, 'Predicted': lstm_predictions_inverse})
results_LSTM


# In[69]:


plt.figure(figsize=(16,6))
plt.plot(y_test, color='blue',label='Actual Distribution of PJME Load')
plt.plot(lstm_predictions, alpha=0.7, color='yellow', label='Distribution of PJME Load')
plt.axhspan(0, 1, facecolor='gray', alpha=0.3)
plt.title("Predictions made by LSTM model")
plt.xlabel('Time')
plt.ylabel('Normalized Distribution of PJME Load')
plt.legend()
plt.show()


# # Compare Predictions
# 
# - For both models ,we see that predicted values are close to the actual values, which means that the actual values, which means that the models perform well in predicting the sequence.
# 

# In[70]:


plt.figure(figsize=(15,8))
plt.plot(y_test, c='black', linewidth=3, label="Original values")
plt.plot(lstm_predictions, c='red',linewidth=3,label="LSTM predictions")
plt.plot(rnn_predictions, alpha=0.5, c="yellow", linewidth=3,label="RNN predictions")
plt.axhspan(0,1, facecolor='gray',alpha=0.3)
plt.legend()
plt.title("Predictions(RNN-LSTM) vs actual data", fontsize=20)
plt.show()


# In[88]:


import pickle
pickle.dump(lstm_predictions_inverse , open('lstm_predictions.pkl','wb'))

model = pickle.load(open('lstm_predictions.pkl','rb'))


# In[83]:


import pickle
pickle.dump(lstm_predictions_inverse, open('lstm_predictions.','wb'))

