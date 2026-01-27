import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ==========================
# CONFIGURACIÓN
# ==========================
MODEL_FOLDER = "modelos_opt"
CSV_PATH = "sipsa.csv"
VENTANA = 24

st.set_page_config(layout="wide")

# ==========================
# HEADER
# ==========================
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
    <p style="margin:0; font-size:1,8rem; font-weight:500;">
        Predicción inteligente de precios de productos agrícolas
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    input::placeholder, textarea::placeholder, select::placeholder {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }
    div[data-baseweb="select"] input {
        color: #ffffff !important;
    }



    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .result-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #a1a1a1;
        margin-bottom: 1rem;
        border-left: 4px solid #ff1a1a;
        padding-left: 1rem;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ==========================
# CARGA DE DATOS
# ==========================
@st.cache_data
def cargar_datos():
    df = pd.read_csv(CSV_PATH, sep=";", decimal=",", encoding="latin-1")
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df.sort_values(["ARTICULO", "FECHA"])
    return df

df = cargar_datos()
productos = df["ARTICULO"].unique()

# ==========================
# LAYOUT EN DOS COLUMNAS
# ==========================
col_left, col_right = st.columns([0.4,0.6])

with col_left:
    st.markdown('<div class="section-title">🛒 Selecciona un producto</div>', unsafe_allow_html=True)
    producto_sel = st.selectbox("Producto:", productos, help='Selecciona el producto para generar la predicción').strip()

df_prod = df[df["ARTICULO"] == producto_sel][["FECHA", "PROMEDIO"]].copy()

# ==========================
# CREAR FEATURES
# ==========================
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

# ==========================
# CARGA DEL MODELO Y SCALERS
# ==========================
@st.cache_resource
def cargar_modelo_scalers(prod):
    modelo = load_model(f"{MODEL_FOLDER}/{prod}_gru_opt.h5")
    scaler_X = joblib.load(f"{MODEL_FOLDER}/{prod}_scaler_X.pkl")
    scaler_y = joblib.load(f"{MODEL_FOLDER}/{prod}_scaler_y.pkl")
    return modelo, scaler_X, scaler_y

modelo, scaler_X, scaler_y = cargar_modelo_scalers(producto_sel)

# ==========================
# ESCALAR FEATURES
# ==========================
X_scaled = scaler_X.transform(df_feat[FEATURES].values)[-VENTANA:]
X_scaled = X_scaled.reshape(1, VENTANA, len(FEATURES))

# ==========================
# PREDICCIÓN
# ==========================
with col_left:
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📆 Selecciona el número de semanas a predecir</div>', unsafe_allow_html=True)
    pred_weeks = st.slider("Número de semanas a predecir:", 1, 12, 4, help='Selecciona el número de semanas que deseas predecir')

preds = []
X_pred = X_scaled.copy()

for i in range(pred_weeks):
    y_pred_scaled = modelo.predict(X_pred)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
    preds.append(y_pred)
    X_pred = np.roll(X_pred, -1, axis=1)
    X_pred[0, -1, 0] = y_pred_scaled[0,0]

# ==========================
# GRÁFICO
# ==========================
historical_dates = df_prod["FECHA"]
historical_prices = df_prod["PROMEDIO"]
future_dates = pd.date_range(df_prod["FECHA"].max(), periods=pred_weeks+1, freq="W")[1:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode="lines", name="Histórico", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines+markers", name="Predicción", line=dict(color="red", dash="dash")))

y_min = min(min(historical_prices), min(preds)) * 0.95
y_max = max(max(historical_prices), max(preds)) * 1.05

fig.update_layout(
    title=dict(
        text=f"Precio histórico vs predicción de {producto_sel}",
        font=dict(size=22, color="black"),
        x=0.25
    ),
    xaxis=dict(
        title=dict(
            text="Fecha",
            font=dict(size=16, color="black")
        ),
        tickfont=dict(size=12, color="black"),
        showgrid=True,
        gridcolor="lightgrey",
        zerolinecolor="lightgrey"
    ),
    yaxis=dict(
        title=dict(
            text="Precio",
            font=dict(size=16, color="black")
        ),
        tickfont=dict(size=12, color="black"),
        showgrid=True,
        gridcolor="lightgrey",
        zerolinecolor="lightgrey"
    ),
    plot_bgcolor="whitesmoke",
    paper_bgcolor="whitesmoke",
    font=dict(color="black"),
    legend=dict(
        font=dict(color="black"),
        bgcolor="rgba(255,255,255,0.7)"
    )
)


# ==========================
# COLUMNA DERECHA → GRÁFICO
# ==========================
with col_right:
    st.markdown('<div class="section-title">📊 Gráfico</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)



# ==========================
# TABLA DE PREDICCIONES
# ==========================
df_pred = pd.DataFrame({"Semana": range(1, pred_weeks+1), "Predicción": preds})

df_pred_clean = df_pred.copy()
df_pred_clean = df_pred_clean[["Semana", "Predicción"]]

# Colores
COLOR_MIN = "color:green; font-weight:bold;"
COLOR_MAX = "color:red; font-weight:bold;"

# Función para colorear toda la fila (no solo la columna de predicción)
def highlight_row(row):
    min_val = df_pred_clean["Predicción"].min()
    max_val = df_pred_clean["Predicción"].max()

    if row["Predicción"] == min_val:
        return [COLOR_MIN, COLOR_MIN]  # Semana + Predicción
    elif row["Predicción"] == max_val:
        return [COLOR_MAX, COLOR_MAX]
    else:
        return ["", ""]

with col_left:
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔮 Predicciones por semana</div>', unsafe_allow_html=True)

    st.dataframe(
        df_pred_clean.style.apply(highlight_row, axis=1),
        use_container_width=True,
        hide_index=True
    )

# ==========================
# MÉTRICAS RÁPIDAS → Basadas en df_pred
# ==========================

with col_right:
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    if df_pred is not None and not df_pred.empty:

        # Valores agregados
        min_pred = df_pred["Predicción"].min()
        mean_pred = df_pred["Predicción"].mean()
        max_pred = df_pred["Predicción"].max()

        # Semanas asociadas
        week_min = df_pred.loc[df_pred["Predicción"].idxmin(), "Semana"]
        week_max = df_pred.loc[df_pred["Predicción"].idxmax(), "Semana"]

        st.markdown('<div class="section-title">🧮 Métricas de predicción</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        # Mínimo
        c1.metric(
            "Predicción mínima",
            f"${min_pred:,.0f}",
            help=f"El mejor momento para comprar según nuestra producción es en la semana {week_min}"
        )
        c1.markdown(
            f"<p style='font-size:1rem; color:green;'>Semana {week_min}</p>",
            unsafe_allow_html=True
        )

        # Promedio
        c2.metric("Predicción promedio", f"${mean_pred:,.0f}")

        # Máximo
        c3.metric(
            "Predicción máxima",
            f"${max_pred:,.0f}",
            help=f"El peor momento para comprar según nuestra producción es en la semana {week_max}"
        )
        c3.markdown(
            f"<p style='font-size:1rem; color:red;'>Semana {week_max}</p>",
            unsafe_allow_html=True
        )

        # Pie de nota
        st.markdown(
            f"<p style='font-size:0.8rem; color:gray;'>"
            f"Cálculos basados en {len(df_pred)} semanas predichas."
            f"</p>",
            unsafe_allow_html=True
        )
