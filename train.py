from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from get_data import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import random

def aria_train(coin):
    # Lấy dữ liệu
    df = get_klines_iter(coin, '1d', '2021-04-27', '2023-04-27')

    # Hàm chuẩn bị dữ liệu cho mô hình ARIMA
    def create_arima_dataset(data, look_back=50):
        dataX, dataY = [], []
        for i in range(len(data) - look_back):
            dataX.append(data[i:(i + look_back)])
            dataY.append(data[i + look_back])
        return np.array(dataX), np.array(dataY)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    look_back = 50
    train_size = int(len(df['Close']) * 0.7)
    train_data, test_data = df['Close'][:train_size], df['Close'][train_size:]
    
    time_line = test_data[look_back:]

    # Tạo tập dữ liệu với định dạng phù hợp cho ARIMA
    trainX, trainY = create_arima_dataset(train_data, look_back)
    testX, testY = create_arima_dataset(test_data, look_back)
    
    # Huấn luyện mô hình ARIMA
    history = [x for x in train_data]
    predictions = list()
    for i in range(len(testY)):
        arima_model = ARIMA(history, order=(5, 1, 0))
        arima_model_fit = arima_model.fit()
        output = arima_model_fit.forecast()
        predictions.append(output[0])
        history.append(testY[i])
          # Thêm giá trị thực tế vào lịch sử để dự đoán bước tiếp theo
    
    # Đánh giá kết quả
    arima_rmse = np.sqrt(mean_squared_error(testY, predictions))
    print("ARIMA RMSE:", arima_rmse)

    history.append(testY[-1])
    arima_model = ARIMA(history, order=(5, 1, 0))
    arima_model_fit = arima_model.fit()
    output = arima_model_fit.forecast()


    fig= go.Figure()
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.01)

    fig.add_trace(go.Scatter(x=time_line.index, 
                            y=testY, 
                            opacity=0.7, 
                            line=dict(color='blue', width=2), 
                            name='Actual'))
    
    fig.add_trace(go.Scatter(x=time_line.index, 
                            y=predictions, 
                            opacity=0.7, 
                            line=dict(color='orange', width=2), 
                            name='Predictions'))
                            
    fig.update_yaxes(title_text="Price ($)")
    fig.update_xaxes(title_text="Time")

    # chart_data.index = df.index
    return fig, arima_rmse, output[0]

def lstm_train(coin):
    df = get_klines_iter(coin, '1d', '2021-04-27', '2023-04-27')

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # Hàm chuẩn bị dữ liệu cho mô hình LSTM
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            dataX.append(dataset[i:(i + look_back)])
            dataY.append(dataset[i + look_back, 3]) # Lấy giá trị Close
        return np.array(dataX), np.array(dataY)

    def create_dataset_for_test(dataset, look_back=1):
        dataX = []
        dataX.append(dataset[len(dataset) - look_back:len(dataset)])
        return np.array(dataX)

    look_back = 50
    time_line = test[look_back:]
    # Chuẩn bị dữ liệu cho LSTM
    
    trainX, trainY = create_dataset(train_scaled, look_back)
    testX, testY = create_dataset(test_scaled, look_back)
    test_1 = create_dataset_for_test(test_scaled, look_back)
    
    # Reshape dữ liệu để phù hợp với đầu vào của LSTM
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))
    test_1 = np.reshape(test_1, (test_1.shape[0], test_1.shape[1], test_1.shape[2]))

    # Xây dựng mô hình LSTM
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')

    # Huấn luyện mô hình LSTM
    lstm_model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

    # Dự đoán kết quả trên tập kiểm tra
    lstm_predictions = lstm_model.predict(testX)
    lstm_predictions = [item for sublist in lstm_predictions for item in sublist]
    
    lstm_predictions_1 = lstm_model.predict(test_1)

    # Đánh giá kết quả
    lstm_rmse = np.sqrt(mean_squared_error(testY, lstm_predictions))
    print("LSTM RMSE:", lstm_rmse)
    fig= go.Figure()
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    
    a = test.values
    mul =  a[-1][3] / testY[-1]

    for i in range(len(lstm_predictions)):
        lstm_predictions[i] = lstm_predictions[i] * mul 
    fig.add_trace(go.Scatter(x=time_line.index,  
                            y = testY * mul ,
                            opacity=0.7, 
                            line=dict(color='red', width=2), 
                            name='Actual'))
    
    
    fig.add_trace(go.Scatter(x=time_line.index, 
                            y = lstm_predictions,
                            opacity=0.7, 
                            line=dict(color='orange', width=2), 
                            name='Predictions'))
                            
    fig.update_yaxes(title_text="Price ($)")
    fig.update_xaxes(title_text="Time")

    # chart_data.index = df.index
    return fig, lstm_rmse, float(lstm_predictions_1[0][0]) * mul