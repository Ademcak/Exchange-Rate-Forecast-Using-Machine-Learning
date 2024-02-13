wwimport pandas as pd
import numpy as np
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from datetime import datetime

# Read data from CSV file
df = pd.read_csv('projekur.csv', parse_dates=['tarih'], dayfirst=True)

# Convert tarih column to a numeric representation of time
def convert_to_timestamp(x):
    if isinstance(x, pd.Timestamp):
        return x.timestamp()
    elif isinstance(x, float):
        return x
    else:
        return datetime.strptime(x, '%d/%m/%Y').timestamp()


df['tarih'] = df['tarih'].apply(convert_to_timestamp)

# Split data into training and test sets
X = df[['tarih', 'dolaralis']]
y = df['dolarsatis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Evaluate the linear regression model
y_pred_lr = model_lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression model performance:")
print(f"RMSE: {rmse_lr}")
print(f"R2 Score: {r2_lr}")

# Fit a random forest regression model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Evaluate the random forest regression model
y_pred_rf = model_rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regression model performance:")
print(f"RMSE: {rmse_rf}")
print(f"R2 Score: {r2_rf}")

# Fit a neural network regression model
model_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model_nn.fit(X_train, y_train)

# Evaluate the neural network regression model
y_pred_nn = model_nn.predict(X_test)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)

print("Neural Network Regression model performance:")
print(f"RMSE: {rmse_nn}")
print(f"R2 Score: {r2_nn}")

""""
# Get the last row of data from the CSV file
last_row = df.iloc[-1]

# Convert the date to a timestamp
next_date = convert_to_timestamp(last_row['tarih']) + 24*60*60

# Create a new row of data for the next day
next_day = pd.DataFrame({'tarih': [next_date], 'dolaralis': [last_row['dolaralis']]})

# Use the linear regression model to predict the exchange rate for the next day
next_day_pred_lr = model_lr.predict(next_day)

print(f"Predicted exchange rate for the next day (using Linear Regression model): {next_day_pred_lr[0]}")
"""

"""
# Convert the date to a timestamp
next_date = datetime.strptime('19/03/2023', '%d/%m/%Y').timestamp()

# Create a new row of data for the next day
next_day = pd.DataFrame({'tarih': [next_date], 'dolaralis': [df['dolaralis'].mean()]})

# Use the linear regression model to predict the exchange rate for the next day
next_day_pred_lr = model_lr.predict(next_day)

print(f"Predicted exchange rate for 19/03/2023 (using Linear Regression model): {next_day_pred_lr[0]}")
"""


