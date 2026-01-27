import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

MODEL_FOLDER = "modelos_opt"
CSV_PATH = "sipsa.csv"
VENTANA = 24

st.set_page_config(layout="wide")

# ================= HEADER =================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #f22727 0%, #f25e2c 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
">
    <h1 style="margin:0; font-size:3rem;">🌾 AgroPrice</h1>
    <p style="margin:0; font-size:1.8rem; font-weight:500;">
        Predicción inteligente de precios de productos agrícolas
    </p>
</div>
""", unsafe_allow_html=True)

# ================= CSS =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] span {
        color: black !important;
        font-weight: 500;
    }

    .result-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.3rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #444;
        margin-bottom: 1rem;
        border-left: 4px solid #ff1a1a;
        padding-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ================= DATOS =================
@st.cache_data
def cargar_datos():
    df = pd.read_csv(CSV_PATH, sep=";", decimal=",", encoding="latin-1")
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df.sort_values(["ARTICULO", "FECHA"])
    return df

df = cargar_datos()
productos = df["ARTICULO"].unique()

col_left, col_right = st.columns([0.4,0.6])

with col_left:
    st.markdown('<div class="section-title">🛒 Selecciona un producto</div>', unsafe_allow_html=True)
    producto_sel = st.selectbox("Producto:", productos).strip()

df_prod = df[df["ARTICULO"] == producto_sel][["FECHA", "PROMEDIO"]].copy()

# ================= FEATURES =================
def crear_features(df):
    df["diff"] = df["PROMEDIO"].diff()
    df["log_return"] = np.log(df["PROMEDIO"] / df["PROMEDIO"].shift(1))
    df["ma_4"] = df["PROMEDIO"].rolling(4).mean()
    df["ma_12"] = df["PROMEDIO"].rolling(12).mean()
    df["vol_4"] = df["PROMEDIO"].rolling(4).std()
    df = df.dropna()
    return df

df_feat = crear_features(df_prod)
FEATURES = ["PROMEDIO", "diff", "log_return", "ma_4", "ma_12", "vol_4"]

@st.cache_resource
def cargar_modelo_scalers(prod):
    modelo = load_model(f"{MODEL_FOLDER}/{prod}_gru_opt.h5")
    scaler_X = joblib.load(f"{MODEL_FOLDER}/{prod}_scaler_X.pkl")
    scaler_y = joblib.load(f"{MODEL_FOLDER}/{prod}_scaler_y.pkl")
    return modelo, scaler_X, scaler_y

modelo, scaler_X, scaler_y = cargar_modelo_scalers(producto_sel)

X_scaled = scaler_X.transform(df_feat[FEATURES].values)[-VENTANA:]
X_scaled = X_scaled.reshape(1, VENTANA, len(FEATURES))

# ================= PREDICCIÓN CORRECTA =================
with col_left:
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📆 Semanas a predecir</div>', unsafe_allow_html=True)
    pred_weeks = st.slider("Número de semanas a predecir:", 1, 12, 4)

preds = []
X_pred = X_scaled.copy()
last_prices = list(df_feat["PROMEDIO"].values[-VENTANA:])

for _ in range(pred_weeks):

    y_pred_scaled = modelo.predict(X_pred, verbose=0)
    y_pred_value = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
    preds.append(y_pred_value)

    last_prices.append(y_pred_value)
    last_prices = last_prices[-VENTANA:]

    temp_df = pd.DataFrame({"PROMEDIO": last_prices})
    temp_df["diff"] = temp_df["PROMEDIO"].diff()
    temp_df["log_return"] = np.log(temp_df["PROMEDIO"] / temp_df["PROMEDIO"].shift(1))
    temp_df["ma_4"] = temp_df["PROMEDIO"].rolling(4).mean()
    temp_df["ma_12"] = temp_df["PROMEDIO"].rolling(12).mean()
    temp_df["vol_4"] = temp_df["PROMEDIO"].rolling(4).std()
    temp_df = temp_df.dropna()

X_feat = temp_df[FEATURES].values

# 🔴 Si aún no hay suficientes filas, rellenamos con la última válida
if len(X_feat) < VENTANA:
    last_row = X_feat[-1]
    padding = np.repeat(last_row.reshape(1, -1), VENTANA - len(X_feat), axis=0)
    X_feat = np.vstack([padding, X_feat])

else:
    X_feat = X_feat[-VENTANA:]

# Escalar
X_new = scaler_X.transform(X_feat)
X_pred = X_new.reshape(1, VENTANA, len(FEATURES))

# ================= GRÁFICO =================
historical_dates = df_prod["FECHA"]
historical_prices = df_prod["PROMEDIO"]
future_dates = pd.date_range(df_prod["FECHA"].max(), periods=pred_weeks+1, freq="W")[1:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode="lines", name="Histórico", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines+markers", name="Predicción", line=dict(color="red", dash="dash")))

fig.update_layout(plot_bgcolor="whitesmoke", paper_bgcolor="whitesmoke", font=dict(color="black"))

with col_right:
    st.markdown('<div class="section-title">📊 Gráfico</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

# ================= TABLA =================
df_pred = pd.DataFrame({"Semana": range(1, pred_weeks+1), "Predicción": preds})

with col_left:
    st.markdown('<div class="section-title">🔮 Predicciones</div>', unsafe_allow_html=True)
    st.dataframe(df_pred, use_container_width=True, hide_index=True)

# ================= MÉTRICAS =================
with col_right:
    st.markdown('<div class="section-title">🧮 Métricas</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Mínimo", f"${min(preds):,.0f}")
    c2.metric("Promedio", f"${np.mean(preds):,.0f}")
    c3.metric("Máximo", f"${max(preds):,.0f}")

