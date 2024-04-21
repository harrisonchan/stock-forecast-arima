import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import models

# from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

TICKER = "AAPL"  # Apple Inc.


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data = load_data(TICKER)
df = data
# print(df.head())
df = df.drop(["Date", "Adj Close"], axis=1)
# print(df.head())

plt.figure(figsize=(12, 6))
plt.plot(df["Close"])
plt.title("TSMC Stock Price")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.grid(True)
plt.show()

# df["Moving Avg 100"] = df["Close"].rolling(100).mean()
# print(df)
moving_avg_100 = df["Close"].rolling(100).mean()
# print(moving_avg_100)

plt.figure(figsize=(12, 6))
plt.plot(df["Close"])
plt.plot(moving_avg_100, "r")
plt.grid(True)
plt.title("Graph Of Moving Averages Of 100 Days")
plt.show()

moving_avg_200 = df["Close"].rolling(200).mean()
# print(moving_avg_200)

# plt.figure(figsize=(12, 6))
# plt.plot(df["Close"])
# plt.plot(moving_avg_100, "r")
# plt.plot(moving_avg_200, "g")
# plt.grid(True)
# plt.title("Comparision Of 100 Days And 200 Days Moving Averages")
# plt.show()

# print(df.shape)

# Splits based on 70% training and 30% testing
split = int(len(data) * 0.7)
training = pd.DataFrame(data[0:split])
testing = pd.DataFrame(data[split:])
# print(training.shape)
# print(testing.shape)
scaler = MinMaxScaler(feature_range=(0, 1))
training_close = training.iloc[:, 4:5].values
testing_close = testing.iloc[:, 4:5].values
data_training_array = scaler.fit_transform(training_close)

x_training = []
y_training = []

for i in range(100, data_training_array.shape[0]):
    x_training.append(data_training_array[i - 100 : i])
    y_training.append(data_training_array[i, 0])

x_training, y_training = np.array(x_training), np.array(y_training)

# model = models.Sequential()
# model.add(
#     LSTM(
#         units=50,
#         activation="relu",
#         return_sequences=True,
#         input_shape=(x_training.shape[1], 1),
#     )
# )
# model.add(Dropout(0.2))


# model.add(LSTM(units=60, activation="relu", return_sequences=True))
# model.add(Dropout(0.3))


# model.add(LSTM(units=80, activation="relu", return_sequences=True))
# model.add(Dropout(0.4))


# model.add(LSTM(units=120, activation="relu"))
# model.add(Dropout(0.5))

# model.add(Dense(units=1))

# model.summary()

# model.compile(
#     optimizer="adam",
#     loss="mean_squared_error",
#     metrics=[tf.keras.metrics.MeanAbsoluteError()],
# )
# model.fit(x_training, y_training, epochs=100)

# model.save(f"{TICKER}.keras")

model = models.load_model("AAPL.keras")
model.summary()

# print("testing_close.shape", testing_close.shape)

past_100_days = pd.DataFrame(training_close[-100:])

# print(past_100_days.shape)

test_df = pd.DataFrame(testing_close)

# print(test_df.shape)

# final_df = past_100_days.append(test_df, ignore_index=True)

final_df = pd.concat([past_100_days, test_df], ignore_index=True)

# print("final_df.head()", final_df.head())

input_data = scaler.fit_transform(final_df)
# print("input_data", input_data)

# print("input_data.shape", input_data.shape)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100 : i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
# print("x_test.shape", x_test.shape)
# print("y_test.shape", y_test.shape)

y_pred = model.predict(x_test)

# print("y_pred.shape", y_pred.shape)

# print("y_test", y_test)

# print("y_pred", y_pred)

scale = scaler.scale_

# print("scaler", scaler)

scale_factor = 1 / scale
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize=(12, 6))
plt.plot(y_test, "b", label="Original Price")
plt.plot(y_pred, "r", label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mae_percentage = (mae / np.mean(y_test)) * 100
print("Mean absolute error on test set: {:.2f}%".format(mae_percentage))

# Actual values
actual = y_test

# Predicted values
predicted = y_pred

# Calculate the R2 score
r2 = r2_score(actual, predicted)

print("R2 score:", r2)

# Plotting the R2 score
fig, ax = plt.subplots()
ax.barh(0, r2, color="skyblue")
ax.set_xlim([-1, 1])
ax.set_yticks([])
ax.set_xlabel("R2 Score")
ax.set_title("R2 Score")

# Adding the R2 score value on the bar
ax.text(r2, 0, f"{r2:.2f}", va="center", color="black")

plt.show()

plt.scatter(actual, predicted)
plt.plot([min(actual), max(actual)], [min(predicted), max(predicted)], "r--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title(f"R2 Score: {r2:.2f}")
plt.show()
