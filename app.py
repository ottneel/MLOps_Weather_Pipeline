import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# 1. Config & Setup
st.set_page_config(page_title="Abuja Weather Forecast", layout="wide")
load_dotenv()

# 2. Database Connection
def get_db_connection():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

# 3. Load Data
@st.cache_data(ttl=600) # Cache data for 10 mins so the DB doesn't get hammered
def load_data():
    engine = get_db_connection()
    
    # A. Get Historical Data (Last 90 days only, to keep graph clean)
    history_query = """
    SELECT date, temp_avg, humidity 
    FROM daily_weather
    ORDER BY date DESC
    LIMIT 90
    """
    df_history = pd.read_sql(history_query, engine)
    df_history['date'] = pd.to_datetime(df_history['date'])
    
    # B. Get Latest Forecast
    # We only want the *most recent* batch of predictions made
    forecast_query = """
    SELECT forecast_date, predicted_temp
    FROM daily_forecasts 
    WHERE created_at = (SELECT MAX(created_at) FROM daily_forecasts)
    ORDER BY forecast_date ASC
    """
    df_forecast = pd.read_sql(forecast_query, engine)
    df_forecast['forecast_date'] = pd.to_datetime(df_forecast['forecast_date'])
    
    return df_history, df_forecast

# 4. The App Layout
try:
    df_history, df_forecast = load_data()

    st.title("🇳🇬 Abuja Weather Predictor")
    st.markdown("### Temperature Forecast")

    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    # Latest actual reading
    current_temp = df_history.iloc[0]['temp_avg']
    current_time = df_history.iloc[0]['date'].strftime('%Y-%m-%d')
    
    col1.metric("Yesterday's Temperature", f"{current_temp:.1f}°C", current_time)
    
    # Next predicted day
    next_pred = df_forecast.iloc[0]['predicted_temp']
    col2.metric("Today's Forecast", f"{next_pred:.3f}°C")

    # Next predicted day
    next_pred = df_forecast.iloc[1]['predicted_temp']
    col3.metric("Tomorrow's Forecast", f"{next_pred:.3f}°C")
    
    # Model Status
    col4.metric("Model Status", "Active")

    # MAIN CHART
    st.subheader("Temperature Trend: History vs. Prediction")
    
    fig = go.Figure()

    # Plot History (Blue Line)
    fig.add_trace(go.Scatter(
        x=df_history['date'], 
        y=df_history['temp_avg'],
        mode='lines',
        name='Historical Data',
        line=dict(color='deepskyblue', width=2)
    ))

    # Plot Forecast (Red Dotted Line)
    fig.add_trace(go.Scatter(
        x=df_forecast['forecast_date'], 
        y=df_forecast['predicted_temp'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='firebrick', width=2, dash='dash')
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLES ---
    with st.expander("See Raw Data"):
        col_a, col_b = st.columns(2)
        col_a.subheader("Recent History")
        col_a.dataframe(df_history.head(10))
        
        col_b.subheader("Upcoming Forecast")
        col_b.dataframe(df_forecast)

except Exception as e:
    st.error(f"Connection Error: {e}")
    st.info("Make sure Docker is running and you have run 'python predict.py' at least once.")