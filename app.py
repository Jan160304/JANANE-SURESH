import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trends Prediction')

user_input= st.text_input('Enter Stock Ticker','AAPL')
df=data.DataReader(user_input,'yahoo',start,end)

st.subheader('Data from 2020-2019')
st.write(df.describe())
