import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import os
import urllib.parse
from dotenv import load_dotenv

# ── CONFIG & SETUP ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Abuja Weather Forecast", layout="wide")
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(override=True)

# ── DATABASE ──────────────────────────────────────────────────────────────────
def get_db_connection():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{urllib.parse.quote_plus(os.getenv('DB_PASS'))}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_data():
    engine = get_db_connection()

    # Historical weather (last 90 days)
    df_history = pd.read_sql("""
        SELECT date, temp_avg, humidity, cloudcover, pressure
        FROM daily_weather
        ORDER BY date DESC
        LIMIT 90
    """, engine)
    df_history['date'] = pd.to_datetime(df_history['date'])

    # Latest temperature forecast
    df_temp = pd.read_sql("""
        SELECT forecast_date, predicted_temp
        FROM daily_forecasts
        WHERE created_at = (SELECT MAX(created_at) FROM daily_forecasts)
        ORDER BY forecast_date ASC
    """, engine)
    df_temp['forecast_date'] = pd.to_datetime(df_temp['forecast_date'])

    # Latest rain forecast — one row per date, most recent prediction wins
    df_rain = pd.read_sql("""
        SELECT DISTINCT ON (forecast_date)
               forecast_date, predicted_rain, rain_probability
        FROM daily_rain_forecasts
        WHERE city = 'Abuja'
        ORDER BY forecast_date DESC, created_at DESC
        LIMIT 7
    """, engine)
    df_rain['forecast_date'] = pd.to_datetime(df_rain['forecast_date'])
    df_rain = df_rain.sort_values('forecast_date')

    return df_history, df_temp, df_rain

# ── APP LAYOUT ────────────────────────────────────────────────────────────────
try:
    df_history, df_temp, df_rain = load_data()

    st.title("Abuja Weather Forecast")

    # ── RAIN SECTION (first) ──────────────────────────────────────────────────
    st.markdown("### Rain Forecast")

    if not df_rain.empty:
        rain_cols = st.columns(min(4, len(df_rain)))
        for i, (col, (_, row)) in enumerate(zip(rain_cols, df_rain.iterrows())):
            label      = row['forecast_date'].strftime('%a %d %b')
            prediction = "Rain" if row['predicted_rain'] == 1 else "No Rain"
            confidence = f"{row['rain_probability']:.0%} confidence"
            col.metric(label, prediction, confidence)

    # Rain probability as line chart
    if not df_rain.empty:
        st.subheader("Rain Probability Trend")
        fig_rain = go.Figure()

        fig_rain.add_trace(go.Scatter(
            x=df_rain['forecast_date'],
            y=df_rain['rain_probability'] * 100,
            mode='lines+markers',
            name='Rain Probability',
            line=dict(color='steelblue', width=2),
            marker=dict(size=8)
        ))

        fig_rain.add_hline(
            y=50, line_dash='dash',
            line_color='white', opacity=0.5,
            annotation_text="50% threshold"
        )

        fig_rain.update_layout(
            xaxis_title="Date",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            template="plotly_dark",
            hovermode="x unified"
        )
        st.plotly_chart(fig_rain, use_container_width=True)

    # ── TEMPERATURE SECTION (second) ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Temperature")
    col1, col2, col3, col4 = st.columns(4)

    current_temp = df_history.iloc[0]['temp_avg']
    current_date = df_history.iloc[0]['date'].strftime('%Y-%m-%d')
    col1.metric("Latest Reading", f"{current_temp:.1f}C", current_date)

    if len(df_temp) >= 1:
        col2.metric("Today's Forecast", f"{df_temp.iloc[0]['predicted_temp']:.1f}C")
    if len(df_temp) >= 2:
        col3.metric("Tomorrow's Forecast", f"{df_temp.iloc[1]['predicted_temp']:.1f}C")

    col4.metric("Model", "Active")

    st.subheader("Temperature: History vs Forecast")
    fig_temp = go.Figure()

    fig_temp.add_trace(go.Scatter(
        x=df_history['date'],
        y=df_history['temp_avg'],
        mode='lines',
        name='Historical',
        line=dict(color='deepskyblue', width=2)
    ))

    fig_temp.add_trace(go.Scatter(
        x=df_temp['forecast_date'],
        y=df_temp['predicted_temp'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='firebrick', width=2, dash='dash')
    ))

    fig_temp.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (C)",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # ── RAW DATA TABLES ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("See Raw Data"):
        col_a, col_b, col_c = st.columns(3)

        col_a.subheader("Recent History")
        col_a.dataframe(df_history.head(10))

        col_b.subheader("Temperature Forecasts")
        col_b.dataframe(df_temp)

        col_c.subheader("Rain Forecasts")
        col_c.dataframe(df_rain)

except Exception as e:
    st.error(f"Connection Error: {e}")
    st.info("Make sure Docker is running and predictions have been generated.")