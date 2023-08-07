import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
from keras.models import load_model 
from keras.layers import  Dense,Dropout,LSTM
from keras.models import Sequential

#background

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('BG6.png')

# APP TITLE

st.title('Stock Prediction app')
user_input = st.text_input('Enter Stock Ticker','TATAPOWER.NS')
hist = yf.Ticker(user_input)
df = hist.history(period = '10y',auto_adjust = "true")
 
#Describing Data
st.write(df.describe())

#Visualizations
st.subheader('closing Price Vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('closing Price Vs Time Chart with 100 Moving Average of '+ ' '+ user_input)
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing Price Vs Time Chart with 100 MA &200MA of '+' '+ user_input)
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

#splitting the data into training and testing set
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler(feature_range = (0,1))
data_training_array = Scaler.fit_transform(data_training)

#load the model
model = load_model('keras_model.h5')    

#Testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = Scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test = np.array(x_test),np.array(y_test)

#prediction
y_predicted = model.predict(x_test)
sc = Scaler.scale_

scale_factor = 1/sc[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Original Vs Predicted price of'+' '+ user_input)
fig2 =plt.figure(figsize =(12,6))
plt.plot(y_test,'b', label = 'original price')
plt.plot(y_predicted, 'r' , label = 'Analysis price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)