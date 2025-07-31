
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

st.set_page_config(page_title="Forecast LSTM KSIP AGRO", layout="wide")

st.title("Forecast LSTM untuk Komoditas KSIP AGRO")
st.markdown("Berikut adalah hasil pemodelan dan prediksi harga komoditas menggunakan LSTM berdasarkan notebook asli.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Load data yang digunakan
df = pd.read_excel('/content/kenya new.xlsx',skiprows=0)

df.head()

df

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# PASTIKAN kolom 'Tanggal' adalah tipe datetime
df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])

# Sortir data berdasarkan tanggal
df = df.sort_values(by='TANGGAL')

# Pilih variabel kolom pada excel untuk proses forecasting
data = df[['TANGGAL', 'HARGA']].set_index('TANGGAL')

# Resampling data ke frekuensi mingguan (jika data harian)
# Jika data sudah mingguan, langkah ini bisa diabaikan atau disesuaikan aja
# Menggunakan 'W' untuk frekuensi mingguan
data_weekly = data['HARGA'].resample('W').mean().dropna().to_frame()

# Normalisasi data (penting untuk model LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_weekly)

# Fungsi untuk membuat dataset time series
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Definisikan look_back (jumlah periode waktu yang digunakan untuk memprediksi periode berikutnya)
look_back = 4 # Contoh: gunakan data 4 minggu sebelumnya untuk memprediksi 1 minggu ke depan

# Buat dataset training dan testing
X, y = create_dataset(scaled_data, look_back)

# Reshape input menjadi [samples, timesteps, features] yang dibutuhkan oleh LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


# Bangun Model LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32, return_sequences=False)) # Explicitly set return_sequences=False for the last LSTM layer
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Latih Model
# Gunakan seluruh data yang ada untuk melatih model sebelum prediksi
model.fit(X, y, epochs=200, batch_size=1, verbose=2)

# Persiapan untuk prediksi 3 minggu berikutnya
# Ambil data 3 minggu terakhir dari data asli yang sudah di-resample untuk digunakan sebagai input prediksi
last_weeks_data = scaled_data[-look_back:].reshape(-1, 1)

# Buat array untuk menyimpan prediksi
forecast = []

# Lakukan prediksi untuk 3 minggu ke depan
for _ in range(4):
    # Reshape data input for prediction (one sample, look_back timesteps, 1 feature)
    input_data = np.reshape(last_weeks_data, (1, look_back, 1))

    # Lakukan prediksi
    predicted_price_scaled = model.predict(input_data, verbose=0) # Add verbose=0 to suppress prediction output

    # Inverse transform prediksi untuk mendapatkan harga sebenarnya
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    # Tambahkan prediksi ke daftar forecast
    forecast.append(predicted_price[0][0])

    # Perbarui 'last_weeks_data' dengan menambahkan prediksi baru dan menghapus data terlama
    # Ini penting untuk prediksi multi-step
    last_weeks_data = np.append(last_weeks_data[1:], [[predicted_price_scaled[0][0]]], axis=0)

# Buat tanggal untuk 4 minggu ke depan
last_date = data_weekly.index[-1]
forecast_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, 5)]

# Tampilkan hasil forecast
print("\nForecast Harga untuk item KENYA di 4 minggu berikutnya:")
for date, price in zip(forecast_dates, forecast):
    print(f"Tanggal: {date.strftime('%Y-%m-%d')}, Estimasi Harga: Rp {price:.2f}")

# Visualisasi
plt.figure(figsize=(12, 6))
plt.plot(data_weekly.index, scaler.inverse_transform(scaled_data), label='Harga Aktual')
plt.plot(forecast_dates, forecast, label='Forecast', marker='o')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.title('Forecast Harga KENYA')
plt.legend()
plt.grid(True)
plt.show()

#akurasi model

# memprediksi data training itu sendiri untuk menghitung metrik
train_predict_scaled = model.predict(X, verbose=0)
train_predict = scaler.inverse_transform(train_predict_scaled)
train_y = scaler.inverse_transform([y]) # Inverse transform actual y for comparison

# Hitung metrik evaluasi pada data training
rmse = np.sqrt(mean_squared_error(train_y[0], train_predict[:,0]))
mae = mean_absolute_error(train_y[0], train_predict[:,0])
r2 = r2_score(train_y[0], train_predict[:,0])

print("\nEvaluasi Model pada Data Training:")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")



import numpy as np
# Menghitung persentase RMSE dan MAE terhadap rata-rata harga aktual pada data training
average_actual_price = np.mean(train_y[0])

rmse_percentage = (rmse / average_actual_price) * 100
mae_percentage = (mae / average_actual_price) * 100

print(f"RMSE dalam persentase terhadap rata-rata harga aktual: {rmse_percentage:.2f}%")
print(f"MAE dalam persentase terhadap rata-rata harga aktual: {mae_percentage:.2f}%")

# plot grafik gabungan

import matplotlib.pyplot as plt
# Tambahkan prediksi pada data training ke plot
# Sesuaikan indeks train_predict agar sesuai dengan tanggal data aktual yang digunakan untuk training
# Indeks data aktual yang digunakan untuk training adalah dari data_kenya_weekly setelah diabaikan look_back periode pertama
train_predict_dates = data_weekly.index[look_back:len(data_weekly.index)]

plt.figure(figsize=(14, 7))

# Plot data aktual
plt.plot(data_weekly.index, scaler.inverse_transform(scaled_data), label='Harga Aktual', color='blue')

# Plot prediksi pada data training
# Pastikan panjang train_predict sesuai dengan panjang train_predict_dates
if len(train_predict_dates) == len(train_predict):
  plt.plot(train_predict_dates, train_predict[:,0], label='Prediksi Training', color='green', linestyle='--')
else:
  print("Warning: Length of train_predict_dates and train_predict do not match. Cannot plot training predictions accurately.")


plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.title('Actual, Training Prediction, and Forecast Harga KENYA')
plt.legend()
plt.grid(True)
plt.show()

#LSTM MULTISTEP

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Jumlah langkah input
n_steps = 30

# Ambil 30 data terakhir sebagai input awal
last_input = scaled_data[-n_steps:]  # Data sudah dinormalisasi dengan scaler
current_input = last_input.reshape(1, n_steps, 1)

# Simpan prediksi
may_predictions = []

# Prediksi 31 hari untuk bulan juli
for i in range(31):
    pred_scaled = model.predict(current_input, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    may_predictions.append(pred_price)

    # Perbarui input (geser window, tambahkan prediksi baru)
    current_input = np.append(current_input[0, 1:, 0], pred_scaled[0][0])
    current_input = current_input.reshape(1, n_steps, 1)

# Tampilkan hasil prediksi
print("Prediksi harga untuk bulan Juli:")
for i, price in enumerate(may_predictions, start=1):
    print(f"Tanggal: 2025-07-{i:02d}, Estimasi Harga: Rp {price:,.2f}")



# plot prediction and training

import pandas as pd
import matplotlib.pyplot as plt

last_date_data = data_weekly.index[-1] # Using the last date from the weekly data
may_forecast_dates = [last_date_data + pd.Timedelta(days=i) for i in range(1, 32)] # 31 days after the last date


plt.figure(figsize=(14, 7))

# Plot data aktual (harian jika ada, atau weekly jika itu yang digunakan)
# Since your data_weekly is weekly, let's plot the weekly data points
plt.plot(data_weekly.index, scaler.inverse_transform(scaled_data), label='Harga Aktual (Weekly Resample)', color='blue', marker='.')


# Plot prediksi pada data training
if len(train_predict_dates) == len(train_predict):
  plt.plot(train_predict_dates, train_predict[:,0], label='Prediksi Training', color='green', linestyle='--')


# Plot prediksi bulan Juli (multi-step LSTM)
plt.plot(may_forecast_dates, may_predictions, label='Forecast (July 2025)', marker='x', color='red', linestyle='-.')

plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.title('Actual, Training Prediction, and Forecast Prices')
plt.legend()
plt.grid(True)
plt.show()


#LSTM linear interpolation (final output)

import pandas as pd
import matplotlib.pyplot as plt
# Karena model dilatih dengan data mingguan, sedangkan tujuan utama adalah untuk memprediksi harga harian maka perlu untuk dilakukan modifikasi proses prediksi
# untuk mendapatkan prediksi harian. :
# 1. Melatih ulang model dengan data harian jika memungkinkan (opsional).
# 2. Menggunakan hasil prediksi mingguan dan mengalokasikannya kembali ke skala harian
#    berdasarkan pola harian historis atau asumsi lainnya (yang digunakan).

# Karena model yang ada dilatih dengan data mingguan (data_weekly),
# cara paling langsung untuk memperluas prediksi adalah dengan:
# a. Memprediksi minggu-minggu berikutnya menggunakan model mingguan.
# b. Mengasumsikan pola harian di setiap minggu yang diprediksi serupa dengan pola harian historis
#    dari minggu-minggu sebelumnya atau menggunakan interpolasi linear.

# Metode 2 (Interpolasi Linear) adalah yang paling mudah diimplementasikan dengan model yang ada.
# Kita akan memprediksi 4 minggu ke depan (seperti sebelumnya) dan kemudian menginterpolasi
# secara linear untuk mendapatkan nilai harian di antara titik-titik prediksi mingguan tersebut.

# Pertama, pastikan Anda memiliki data harian asli (sebelum resampling)
# Gunakan 'data' yang sudah di-filter ['TANGGAL', 'PRICE']
data_daily = df[['TANGGAL', 'HARGA']].set_index('TANGGAL')

# Normalisasi data harian untuk perhitungan nanti (meskipun tidak digunakan langsung untuk melatih model ini)
scaler_daily = MinMaxScaler(feature_range=(0, 1))
scaled_data_daily = scaler_daily.fit_transform(data_daily)


# Kita sudah memprediksi 4 titik mingguan ke depan (forecast_dates dan forecast)
# forecast_dates: Tanggal awal minggu yang diprediksi (dari minggu ke-1 hingga minggu ke-4 setelah data terakhir)
# forecast: Nilai prediksi untuk minggu-minggu tersebut

# Sekarang, kita akan membuat tanggal harian untuk satu bulan ke depan (sekitar 30 hari)
last_date_actual_daily = data_daily.index[-1]
forecast_start_date = last_date_actual_daily + pd.Timedelta(days=1)
forecast_end_date = forecast_start_date + pd.Timedelta(days=30) # Prediksi 31 hari untuk 1 bulan

# Buat rentang tanggal harian untuk satu bulan ke depan
forecast_dates_daily = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')

# Untuk mendapatkan prediksi harian, kita bisa melakukan interpolasi antara titik-titik prediksi mingguan.
# Titik-titik yang diketahui untuk interpolasi adalah:
# - Titik data aktual terakhir (harga aktual pada tanggal terakhir data_daily)
# - Titik-titik prediksi mingguan (pada tanggal forecast_dates dengan nilai forecast)

# Gabungkan data aktual terakhir dan prediksi mingguan untuk interpolasi
# Kita perlu menggabungkan tanggal dan nilai
interpolation_dates = [data_daily.index[-1]] + forecast_dates
interpolation_values = [data_daily['HARGA'].iloc[-1]] + forecast

# Buat Series Pandas untuk interpolasi
interpolation_series = pd.Series(interpolation_values, index=interpolation_dates)

# Lakukan interpolasi linear pada rentang tanggal harian yang diinginkan
# Reindex untuk memasukkan tanggal harian yang diprediksi, lalu interpolasi
forecast_daily_interpolated = interpolation_series.reindex(interpolation_series.index.union(forecast_dates_daily)).interpolate(method='time').loc[forecast_dates_daily]

# Tampilkan hasil prediksi harian
print("\nForecast Harga Harian untuk satu bulan kedepan:")
for date, price in forecast_daily_interpolated.items():
    print(f"Tanggal: {date.strftime('%Y-%m-%d')}, Estimasi Harga: Rp {price:.2f}")

# Visualisasi gabungan: data aktual harian, prediksi mingguan, dan prediksi harian hasil interpolasi

plt.figure(figsize=(15, 8))

# Plot data aktual harian
plt.plot(data_daily.index, data_daily['HARGA'], label='Harga Aktual Harian', color='blue', alpha=0.7)

# Plot titik prediksi mingguan (untuk referensi)
plt.plot(forecast_dates, forecast, label='Prediksi Mingguan (Hasil Model)', marker='o', linestyle='', color='red', markersize=6)

# Plot prediksi harian hasil interpolasi
plt.plot(forecast_dates_daily, forecast_daily_interpolated, label='Forecast Harian (Interpolasi)', color='orange', linestyle='--')


plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.title('Actual, Training Prediction, and Forecast Harga KENYA')
plt.legend()
plt.grid(True)
plt.show()

#NOTE : jika data historis sudah tergolong banyak (misal 6 tahun)
#       tidak perlu lagi pake interpolasi linear, bisa lgsg latih model menggunakan
#       data harian.
