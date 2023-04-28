import datetime as dt
import numpy as np
import pandas as pd
from streamlit_plotly_events import plotly_events
from multiprocessing import Process
from get_data import get_data
from binance.client import Client
from time import sleep
from streamlit_option_menu import option_menu
import streamlit as st
import json
import pandas as pd
from plot_graph import plot_graph, plot_percent
from train import *
# st.set_page_config(page_title="Crypto Dashboard")

def round_value(input_value):
        if input_value.values > 1:
            a = float(round(input_value, 2))
        else:
            a = float(round(input_value, 8))
        return a

crpytoList = ('BTCBUSD', 'ETHBUSD', 'BNBBUSD', 'XRPBUSD', 'ADABUSD', 'DOGEBUSD', 'SHIBBUSD', 'DOTBUSD')
    

column_1 = ['BTCBUSD', 'ETHBUSD', 'BNBBUSD']
column_2 = ['XRPBUSD', 'ADABUSD', 'DOGEBUSD']
column_3 = ['SHIBBUSD', 'DOTBUSD']


def dashboard():
    st.markdown("""# **Crypto Dashboard** """)
    st.markdown("""A simple cryptocurrency price app pulling price data from the [Binance API](https://www.binance.com/en/support/faq/360002502072)""")
    coin = st.selectbox("Cryptocurrencies", crpytoList) 
    col1, col2, col3 = st.columns(3)
    placeholder = st.empty()

    for _ in range(1000):
        df = pd.read_json("https://api.binance.com/api/v3/ticker/24hr")
        
        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                for i in column_1:
                    col_df = df[df.symbol == i] # dataframe of the coin 
                    col_price = round_value(col_df.weightedAvgPrice)
                    col_percent = f"{float(col_df.priceChangePercent)}%"
                    st.metric(i, col_price, col_percent)
            with col2:
                for i in column_2:
                    col_df = df[df.symbol == i]
                    col_price = round_value(col_df.weightedAvgPrice)
                    col_percent = f"{float(col_df.priceChangePercent)}%"
                    st.metric(i, col_price, col_percent)
            with col3:
                for i in column_3:
                    col_df = df[df.symbol == i]
                    col_price = round_value(col_df.weightedAvgPrice)
                    col_percent = f"{float(col_df.priceChangePercent)}%"
                    st.metric(i, col_price, col_percent)
            st.header("")

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            download_csv = df[["symbol", "openPrice", "highPrice", "lowPrice", "volume", "priceChangePercent"]]
            download_csv.columns = ["Cryptocurrency", "Open", "High", "Low", "Volume" , "Change(%)"]
            csv = convert_df(download_csv)
            st.dataframe(download_csv)
            st.download_button(
                    label="Download market's data as CSV",
                    data=csv,
                    file_name="Market_data.csv",
                    mime="text/csv",
                )
            
            data_symbol = download_csv[download_csv.Cryptocurrency == coin]
            chart_data = get_data(coin, 30)
            st.header(coin)
            st.dataframe(chart_data)
            csv = convert_df(chart_data)
            st.download_button(
                    label=f"Download {coin} data as CSV",
                    data=csv,
                    file_name=f"{coin}_data.csv",
                    mime="text/csv",
                )

            fig = plot_graph(chart_data)
            st.plotly_chart(fig)
            
            percent = plot_percent(chart_data, data_symbol)

            st.plotly_chart(percent)
            
            sleep(10)

def algorithm():
    al = st.selectbox("Algorithm", ["ARIMA", "LSTM"])

    coin = st.selectbox("Cryptocurrencies", crpytoList) 

    train = st.button("Train")

    if train:
        st.write(f"{al} Training with {coin} data")
        if al == "ARIMA":
            with st.spinner("Training..."):
                fig, rmse, tomorrow = aria_train(coin)

                st.plotly_chart(fig)

                st.write(f"RMSE: {rmse}")

                st.write(f"Tomorrow price: {tomorrow}") 

        elif al == "LSTM":
            with st.spinner("Training..."):
                fig, rmse, tomorrow = lstm_train(coin)

                st.plotly_chart(fig)

                st.write(f"RMSE: {rmse}")

                st.write(f"Tomorrow price: {tomorrow}")       

selected = option_menu(
            menu_title=None,  # required
            options=["Dashboard", "Algorithm"],  # required
            icons=["house", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#E8A0BF"},
            },
        )

if selected == "Dashboard":
    dashboard()

elif selected == "Algorithm":
    algorithm()
