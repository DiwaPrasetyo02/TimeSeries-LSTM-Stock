# -*- coding: utf-8 -*-
"""Submission_Diwa_Time_Series.ipynb


# Import Library
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

"""# Data Preprocessing"""

df = pd.read_csv('/content/drive/MyDrive/dataset/Sunspots.csv')
df.rename(columns = {'Monthly Mean Total Sunspot Number':'mean_sun'}, inplace = True)
df

df.shape
# 1. Data Memiliki 3265 sample, sudah memenuhi kriteria 1

# Normalisasi data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['mean_sun']])

"""# Modeling"""

# Membuat data time series
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)

# Membagi data menjadi train dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #3. Validation test sebesar 0.2

# Membentuk ilang intputan train, time steps yang dibutuhkan untuk LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 2. Membuat model LSTM
model = Sequential() #4. menggunakan model sequential
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.001) #5. menggunakan learning rate
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Early stopping ( sama seperti callbacks )
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

# Evaluasi model
mae = model.evaluate(X_test, y_test, verbose=0)[1]
print(f'MAE: {mae}')

# Skala data maksimum
scale_max = scaled_data.max()
threshold = 0.10 * scale_max

if mae < threshold:
    print(f"MAE {mae} memenuhi kriteria <10% dari skala data.")
else:
    print(f"MAE {mae} tidak memenuhi kriteria <10% dari skala data.")

#6. Mae < 10% skala data

# Plot Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.show()
