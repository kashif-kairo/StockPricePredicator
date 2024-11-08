import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

Start = "2015-01-01"
Today= date.today().strftime("%y-%m-%d")
Today='20'+Today
st.title("Stock Prediction Web APP by Ahmad")

stocks=("AAPL","GOOG","MSFT","GME","ITC")

selected_stock=st.selectbox("Select dataset for prediction",stocks)
n_years=st.slider("Years of prediction: ", 1, 4)
period=n_years*365

# @st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, Start, Today)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

data_load_state=st.text("Load data.....")
data=load_data(selected_stock)
data_load_state.text("Loading data ...done!")

st.subheader('RAW DATA')
st.write(data.tail())

def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name="stock_close"))
    fig.layout.update(title_text="Times Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()
#Forecasting
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

st.subheader('FORECAST DATA')
st.write(forecast.tail())
st.write('FORECAST DATA')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)
fig2=m.plot_components(forecast)
st.write(fig2)



