import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import seaborn as sns
# Define file path
file_path = r'C:\Users\appua\Desktop\Python\python ptoject\NSE India listed companiesAMRUTANJAN_NS.csv'

# Title for the web app
st.title('Stock Trends Prediction')
user_input= st.text_input('Enter Stock Ticker','AMUR')
# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Filter data for the date range 2014-01-01 to 2024-12-31
df = df[(df['Date'] >= '2014-01-01') & (df['Date'] <= '2024-12-31')]

# Display filtered data description
st.subheader('Data Description (2014-2024)')
st.write(df.describe())

# Allow user to select columns for visualization
columns = st.multiselect('Select Columns to Visualize', df.columns.tolist(), default=['Open', 'Close'])

# Display the selected columns as a plot
if columns:
    st.subheader('Selected Columns Visualization')
    st.line_chart(df.set_index('Date')[columns])  # Use 'Date' as the index for plotting
else:
    st.write('Please select columns to visualize.')


# Load the dataset
file_path = r'C:\Users\appua\Desktop\Python\python ptoject\NSE India listed companiesAMRUTANJAN_NS.csv'
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter the dataset for the required date range (2014-01-01 to 2024-12-31)
filtered_df = df[(df['Date'] >= '2014-01-01') & (df['Date'] <= '2024-12-31')]

# Closing Price Trend Over Time
st.title("Stock Trends Prediction Web App")
st.subheader("Closing Price vs Time Chart")
plt.figure(figsize=(12, 6))
plt.plot(filtered_df['Date'], filtered_df['Close'], color='blue', label='Closing Price')
plt.title("Closing Price Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.grid()
st.pyplot(plt)  # Render the plot in Streamlit

# Calculate daily percentage change
filtered_df['Daily Change'] = filtered_df['Close'].pct_change()

# Plot the histogram of daily percentage change
st.subheader("Distribution of Daily Percentage Changes")
plt.figure(figsize=(10, 6))
sns.histplot(filtered_df['Daily Change'].dropna(), bins=50, kde=True, color="skyblue")
plt.title("Distribution of Daily Percentage Changes")
plt.xlabel("Daily Percentage Change")
plt.ylabel("Frequency")
plt.grid()
st.pyplot(plt)

# Calculate rolling volatility (20-day standard deviation)
filtered_df['Rolling Volatility'] = filtered_df['Close'].rolling(window=20).std()

# Plot Rolling Volatility over Time
st.subheader("Rolling Volatility of Closing Prices")
plt.figure(figsize=(12, 6))
plt.plot(filtered_df['Date'], filtered_df['Rolling Volatility'], color='red', label='Rolling Volatility (20-day)')
plt.title("Rolling Volatility of Closing Prices")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid()
st.pyplot(plt)

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
plt.figure(figsize=(10, 8))
sns.heatmap(filtered_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
st.pyplot(plt)

st.subheader('Closing Price vs Time Chart')
fig =plt.figure(figsize=(10,5))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling (100).mean()
plt.plot(ma100,'orange')
plt.plot(df.Close,'Green')
st.pyplot(fig)

import matplotlib.pyplot as plt
import streamlit as st

# Add a subheader for the visualization
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')

# Calculate the 100-day and 200-day moving averages
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

# Create a new figure for plotting
fig = plt.figure(figsize=(12, 6))  # Define figure size

# Plot the 100-day moving average in orange
plt.plot(ma100, 'orange', label='100-Day Moving Average')

# Plot the 200-day moving average in red
plt.plot(ma200, 'red', label='200-Day Moving Average')

# Plot the closing price in green
plt.plot(df['Close'], 'green', label='Closing Price')

# Add labels, legend, and grid
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.grid()

# Display the plot in the Streamlit app
st.pyplot(fig)



data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])  # 70% of the data for training
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])  # Remaining 30% for testing

from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler with the correct feature_range tuple
scaler = MinMaxScaler(feature_range=(0, 1))

model=load_model('renamed_model.keras')
#testing
past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test= np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)
Scaler=scaler.scale_
scale_factor=1/Scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor


#final graph
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)