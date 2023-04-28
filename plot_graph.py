import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pandas as pd

def plot_graph(df):
        fig= go.Figure()
       
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.7, 0.3])
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
      

        fig.add_trace(go.Candlestick(x=df.index, 
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'], name='market data'))

        fig.add_trace(go.Scatter(x=df.index, 
                                y=df['MA5'], 
                                opacity=0.7, 
                                line=dict(color='blue', width=2), 
                                name='MA 5'))

        fig.add_trace(go.Scatter(x=df.index, 
                                y=df['MA20'], 
                                opacity=0.7, 
                                line=dict(color='orange', width=2), 
                                name='MA 20'))

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        colors = ['green' if row['Open'] - row['Close'] >= 0 
                else 'red' for index, row in df.iterrows()]

        fig.add_trace(go.Bar(x=df.index, 
                        y=df['Volume'],
                        marker_color=colors
                        ), row=2, col=1)

        return fig

def plot_percent(df, data_symbol):
        fig= go.Figure()
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.01)

        X = df.Close.values
        X_open = data_symbol.Open.values

        percent = []
        for i in range(len(X)):
                percent.append(100 * (X[i] - X_open[0])/ X_open[0])  

        chart_data = pd.DataFrame(
                percent,
                columns=['Percent (%)'], index=df.index.copy())     

        fig.add_trace(go.Scatter(x=chart_data.index, 
                                y=chart_data['Percent (%)'], 
                                opacity=0.7, 
                                line=dict(color='blue', width=2), 
                                name='MA 5'))
                                
        fig.update_yaxes(title_text="Percent (%)")
        fig.update_xaxes(title_text="Time")

        # chart_data.index = df.index
        return fig