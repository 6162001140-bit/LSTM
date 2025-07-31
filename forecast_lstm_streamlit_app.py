import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.title("LSTM Forecast Harga Mingguan")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Data Awal")
    st.write(df.head())

    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    df = df.sort_values(by='TANGGAL')
    data = df[['TANGGAL', 'HARGA']].set_index('TANGGAL')
    data_weekly = data['HARGA'].resample('W').mean().dropna().to_frame()

    st.line_chart(data_weekly)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_weekly)

    def create_dataset(dataset, look_back=4):
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:i+look_back, 0])
            y.append(dataset[i+look_back, 0])
        return np.array(X), np.array(y)

    look_back = 4
    X, y = create_dataset(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=1, verbose=0)

    last_data = scaled_data[-look_back:]
    future_preds = []
    input_seq = last_data.reshape(1, look_back, 1)

    for _ in range(4):
        pred = model.predict(input_seq, verbose=0)
        future_preds.append(pred[0][0])
        input_seq = np.append(input_seq[:,1:,:], [[pred[0]]], axis=1)

    forecast = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    st.subheader("Forecast Minggu Depan (4 minggu)")
    st.line_chart(pd.DataFrame(forecast, columns=["Forecast"]))
# Buat tanggal untuk 4 minggu ke depan
last_date = data_weekly.index[-1]
forecast_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, 5)]

# Tabel hasil prediksi
forecast_df = pd.DataFrame({
    "Tanggal": forecast_dates,
    "Harga Prediksi": forecast.flatten()
})

st.subheader("Tabel Forecast Harga")
st.dataframe(forecast_df)

# Visualisasi gabungan data historis + forecast
full_plot = pd.concat([
    data_weekly.rename(columns={"HARGA": "Harga Aktual"}),
    forecast_df.set_index("Tanggal").rename(columns={"Harga Prediksi": "Harga Aktual"})
])

st.subheader("Visualisasi Harga (Aktual + Forecast)")
st.line_chart(full_plot)

