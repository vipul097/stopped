import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from datetime import date
st.title('Stock Prediction Using LSTM')
ticker = st.sidebar.text_input('Ticker','IBN')
start_date= st.sidebar.date_input('Start Date')
end_date=st.sidebar.date_input('End Date')
df=yf.download(ticker,start=start_date,end=end_date)
df
fig=px.line(df, x=df.index, y=df['Close'], title=ticker)
st.plotly_chart(fig)

df1=df[['Close']]
df1['%Change']=df1['Close']/df1['Close'].shift(1)-1
df1.dropna(inplace=True)
df1['log returns']=np.log(1+df1['%Change'])
df1.dropna(inplace=True)
X=df1[['Close','log returns']].values
scaler=MinMaxScaler(feature_range=(0,1)).fit(X)
X_scaled=scaler.transform(X)
y=[x[0] for x in X_scaled]
X_train=X_scaled
y_train=y

n=3
Xtrain=[]
ytrain=[]
for i in range(n, len(X_train)):
    Xtrain.append(X_train[i-n:i,:X_train.shape[1]])
    ytrain.append(y_train[i])

Xtrain,ytrain=(np.array(Xtrain),np.array(ytrain))
Xtrain=np.reshape(Xtrain,(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

from keras.models import Sequential
from keras.layers import Dense, LSTM

model=Sequential()
model.add(LSTM(4, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(Xtrain,ytrain,epochs=100,batch_size=16,verbose=1)

trainpredict=model.predict(Xtrain)
trainpredict=np.c_[trainpredict,np.zeros(trainpredict.shape)]

tp=scaler.inverse_transform(trainpredict)
tp=[x[0] for x in tp]

st.header('Table showing Closing price and Predicted Price')
close_train=df.Close[4:]
close_df=pd.DataFrame(close_train)
#close_df['Close']=close_df['Close'].shift(1)
close_df['Predicted Price']=tp
#close_df['Predicted Price']=close_df['Predicted Price'].shift(1)
st.write(close_df)



st.header("Orginal Vs Predicted Price")
fig2=plt.figure(figsize=(8,6))
plt.plot(close_df['Close'], color='green', label='Orginal Price')
plt.plot(close_df['Predicted Price'], color='red', label='Predicted Price')
plt.legend()
plt.show()
st.plotly_chart(fig2)

st.header("Predicted Next Day Price")
df23=df1.copy()
df23=df23[['Close','log returns']]
def min_max_scaler(columns):
    X_min=columns.min()
    X_max=columns.max()
    return (columns-X_min)/(X_max-X_min)

scaled_columns=['Close','log returns']
df23[scaled_columns]=df23[scaled_columns].apply(min_max_scaler)
X_tom=df23[len(df23)-12:]
X_tom=X_tom.values
n=3
tomXtrain=[]
for i in range(n, len(X_tom)):
    tomXtrain.append(X_tom[i-n:i,:X_tom.shape[1]])

tomXtrain=(np.array(tomXtrain))
tomXtrain=np.reshape(tomXtrain,(tomXtrain.shape[0], tomXtrain.shape[1], tomXtrain.shape[2]))

tomtrainpredict=model.predict(tomXtrain)
tomtrainpredict=np.c_[tomtrainpredict,np.zeros(tomtrainpredict.shape)]
tomtp=scaler.inverse_transform(tomtrainpredict)
tomtp=[x[0] for x in tomtp]
st.write(tomtp[8])
