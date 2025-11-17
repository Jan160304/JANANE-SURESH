import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data

import yfinance as yf

start = '2010-01-01'
end = '2019-12-31'
df = yf.download('AAPL', start=start, end=end)
print(df.head())

# Filter numeric columns for statistical operations
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Basic Statistics
mean_values = numeric_df.mean()
median_values = numeric_df.median()
std_values = numeric_df.std()
variance_values = numeric_df.var()

print("Mean Values:\n", mean_values)
print("\nMedian Values:\n", median_values)
print("\nStandard Deviation:\n", std_values)
print("\nVariance:\n", variance_values)

# Mode (special handling as it can return multiple rows)
mode_values = numeric_df.mode()
print("\nMode Values:\n", mode_values.iloc[0])  # Picking first row of mode

# Fill missing data with mean
df_filled_mean = df.fillna(df.mean())

# Fill missing data with the mode (most frequent value)
df_filled_mode = df.fillna(df.mode().iloc[0])

# Print the first few rows of the mode-filled DataFrame
print(df_filled_mode.head())

# Fill missing data with a specific value
df_filled_custom = df.fillna(0)  # Replace missing values with 0
print(df_filled_mean.head())

# Drop rows with missing values
df_dropped_rows = df.dropna()

# Drop columns with missing values
df_dropped_columns = df.dropna(axis=1)

print(df_dropped_rows.head())
print(df_dropped_columns.head())

# Interpolate missing data (linear method)
df_interpolated = df.interpolate(method='linear')

# Interpolation visualized
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Original')
plt.plot(df_interpolated['Close'], label='Interpolated', linestyle='dashed')
plt.title("Interpolation of Missing Data")
plt.legend()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

# Load dataset
start = '2010-01-01'
end = '2019-12-31'
df = yf.download('AAPL', start=start, end=end)

# Verify column names and handle MultiIndex
print("Columns in the dataset:\n", df.columns)

# Flatten MultiIndex (if applicable)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[1] if col[1] else col[0] for col in df.columns]

# Check if 'Close' column exists
if 'Close' in df.columns:
    # Ensure 'Close' column is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # Convert non-numeric values to NaN

    # Handle missing values
    df = df.dropna(subset=['Close'])  # Drop rows with missing values in 'Close'

    # Alternatively, fill missing values with mean
    # df['Close'] = df['Close'].fillna(df['Close'].mean())

    # Generate the box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['Close'], color='skyblue')
    plt.title("Box Plot of Closing Prices")
    plt.show()
else:
    print("'Close' column not found in the dataset.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
import yfinance as yf
start = '2010-01-01'
end = '2019-12-31'
df = yf.download('AAPL', start=start, end=end)

# Basic statistics
print("Dataset Overview:\n")
print(df.head())
print("\nSummary Statistics:\n")
print(df.describe())

# Visualizing closing price trends
plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.title('AAPL Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.grid()
plt.show()

# Correlation analysis
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of daily price changes
df['Daily Change'] = df['Close'].pct_change()
plt.figure(figsize=(12, 6))
plt.hist(df['Daily Change'].dropna(), bins=50, color='skyblue')
plt.title('Distribution of Daily Percentage Change')
plt.xlabel('Daily Change')
plt.ylabel('Frequency')
plt.show()

# Resetting index to flatten the MultiIndex
df = df.reset_index()

# Confirm available column names
print(df.columns)

# Rolling volatility on 'Close' prices
rolling_std = df['Close'].rolling(window=20).std()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], rolling_std, label='Rolling Volatility (20-day window)')
plt.title('Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid()
plt.show()

df.tail()

df=df.reset_index()
df.head()

df = df.drop(['Date', 'Adj Close'], axis=1, errors='ignore')
df.head()

plt.plot(df.Close)


df

ma100=df.Close.rolling(100).mean()
ma100

plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')

ma200=df.Close.rolling(200).mean()
ma200

plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')


df.shape

#spliting

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)

data_training.head()

data_testing.head()

from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler with the correct feature_range tuple
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array=scaler.fit_transform(data_training)
data_training_array

data_training_array.shape

x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)

from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential


from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout

model = Sequential()
model.add(Input(shape=(100, 1)))  # Input layer specifying timesteps and features
model.add(LSTM(50, return_sequences=True))  # Output sequences for stacking
model.add(Dropout(0.2))
model.add(LSTM(60, return_sequences=True))  # Output sequences for stacking
model.add(Dropout(0.3))
model.add(LSTM(80, return_sequences=True))  # Output sequences for stacking
model.add(Dropout(0.4))
model.add(LSTM(120))  # Final LSTM without return_sequences
model.add(Dropout(0.5))
model.add(Dense(1))  # Fully connected layer

model.compile(optimizer='adam', loss='mean_squared_error')

print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)

# Save the model in native Keras format
model.save('my_model.keras')

# Alternatively, you can use the full function call
from keras.saving import save_model
save_model(model, 'my_model.keras')

data_testing.head()

data_training.tail(100)

past_100_days=data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

final_df.head()

input_data=scaler.fit_transform(final_df)
input_data

input_data.shape

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test= np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)

y_predicted=model.predict(x_test)

y_predicted.shape

y_test

y_predicted

scaler.scale_

scale_factor=1/0.02099517
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()



