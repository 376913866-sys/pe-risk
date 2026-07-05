import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils.preprocess import calc_hsi, safe_dataframe, format_ga

# =========================
# 页面
# =========================
st.set_page_config(page_title="PE Prediction System", layout="wide")

# =========================
# 模型加载（缓存防崩）
# =========================
@st.cache_resource
def load_models():
    pe = joblib.load("models/PE_model.pkl")
    early = joblib.load("models/Early_PE_model.pkl")
    pdo = joblib.load("models/PDO_model.pkl")
    ga = joblib.load("models/GA_model.pkl")
    features = joblib.load("models/features.pkl")
    return pe, early, pdo, ga, features

PE_MODEL, EARLY_MODEL, PDO_MODEL, GA_MODEL, FEATURES = load_models()

# =========================
# UI
# =========================
st.title("🩺 子痫前期预测系统（Stable Production Version）")

st.warning("仅科研用途")

# =========================
# 输入
# =========================
age = st.number_input("年龄", 15, 60, 30)
bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
parity = st.number_input("产次", 0, 10, 0)

pe_history = st.selectbox("PE既往史", [0, 1])
htn = st.selectbox("高血压", [0, 1])
dm = st.selectbox("糖尿病", [0, 1])

pappa = st.number_input("MoM PAPP-A", 0.1, 5.0, 1.0)
pi = st.number_input("MoM PI", 0.1, 5.0, 1.0)
map_v = st.number_input("MoM MAP", 0.1, 5.0, 1.0)

plt = st.number_input("Platelet", 50.0, 1000.0, 250.0)

ast = st.number_input("AST", 1.0, 200.0, 20.0)
alt = st.number_input("ALT", 1.0, 200.0, 20.0)

efw = st.number_input("EFW Percentile", 0.0, 100.0, 50.0)

hsi = calc_hsi(ast, alt, bmi)
st.metric("HSI", f"{hsi:.2f}")

# =========================
# 预测
# =========================
if st.button("🚀 Predict"):

    X = pd.DataFrame([{
        "age": age,
        "BMI": bmi,
        "parity": parity,
        "子痫前期既往史": pe_history,
        "慢性高血压": htn,
        "糖尿病": dm,
        "MoM_PAPP-A": pappa,
        "MoM_PI": pi,
        "MoM_MAP": map_v,
        "Plt": plt,
        "HSI": hsi,
        "EFW_percentile": efw
    }])

    X = safe_dataframe(X, FEATURES)

    pe = PE_MODEL.predict_proba(X)[0, 1]
    early = EARLY_MODEL.predict_proba(X)[0, 1]
    pdo = PDO_MODEL.predict_proba(X)[0, 1]
    ga = GA_MODEL.predict(X)[0]

    st.divider()
    st.subheader("📊 Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("PE", f"{pe*100:.1f}%")
    col2.metric("Early PE", f"{early*100:.1f}%")
    col3.metric("PDO", f"{pdo*100:.1f}%")
    col4.metric("GA", format_ga(ga))
