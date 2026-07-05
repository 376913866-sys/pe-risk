import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# 页面设置
# =========================
st.set_page_config(
    page_title="PE Risk Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =========================
# 模型加载（稳定版）
# =========================
@st.cache_resource
def load_models():
    pe = joblib.load("PE_model.pkl")
    early = joblib.load("Early_PE_model.pkl")
    pdo = joblib.load("PDO_model.pkl")
    ga = joblib.load("GA_model.pkl")
    features = joblib.load("features.pkl")
    return pe, early, pdo, ga, features

PE_MODEL, EARLY_MODEL, PDO_MODEL, GA_MODEL, FEATURES = load_models()

# =========================
# 工具函数
# =========================
def parse_hsi(ast, alt, bmi):
    try:
        if ast <= 0 or alt <= 0:
            return 0.0
        return 8 * (alt / ast) + bmi + 2
    except:
        return 0.0


def risk_level(p):
    p = float(np.nan_to_num(p, nan=0.0))
    if p < 0.05:
        return "🟢 低风险"
    elif p < 0.15:
        return "🟡 中风险"
    else:
        return "🔴 高风险"


def ga_format(x):
    try:
        w = int(x)
        d = round((x - w) * 7)
        return f"{w}+{d}"
    except:
        return "N/A"


def safe_predict_proba(model, X):
    try:
        return float(model.predict_proba(X)[0, 1])
    except:
        return float(model.predict(X)[0])


# =========================
# UI
# =========================
st.title("🩺 子痫前期风险预测系统（PE Prediction System）")

st.warning("⚠️ 仅用于科研与教学，不用于临床决策")

# =========================
# 输入区
# =========================
st.header("① 基本信息")

c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("年龄", 15, 60, 30)
with c2:
    bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
with c3:
    parity = st.number_input("产次", 0, 10, 0)

st.header("② 高危因素")

c1, c2, c3 = st.columns(3)
with c1:
    pe_history = st.selectbox("子痫前期既往史", [0, 1])
with c2:
    htn = st.selectbox("慢性高血压", [0, 1])
with c3:
    dm = st.selectbox("糖尿病", [0, 1])

st.header("③ FMF指标")

c1, c2, c3 = st.columns(3)
with c1:
    pappa = st.number_input("MoM PAPP-A", 0.1, 5.0, 1.0)
with c2:
    pi = st.number_input("MoM PI", 0.1, 5.0, 1.0)
with c3:
    map_v = st.number_input("MoM MAP", 0.1, 5.0, 1.0)

st.header("④ 实验室指标")

plt = st.number_input("Platelet", 50.0, 1000.0, 250.0)

st.header("⑤ 肝功能 + HSI")

c1, c2 = st.columns(2)
with c1:
    ast = st.number_input("AST", 1.0, 200.0, 20.0)
with c2:
    alt = st.number_input("ALT", 1.0, 200.0, 20.0)

hsi = parse_hsi(ast, alt, bmi)
st.metric("HSI", f"{hsi:.2f}")

st.header("⑥ 超声指标")

efw = st.number_input("EFW Percentile", 0.0, 100.0, 50.0)

# =========================
# 预测按钮
# =========================
if st.button("🚀 开始预测"):

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

    # 对齐特征
    X = X.reindex(columns=FEATURES)

    # 处理异常
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    X = X.astype(np.float32)

    # =========================
    # 预测
    # =========================
    pe = safe_predict_proba(PE_MODEL, X)
    early = safe_predict_proba(EARLY_MODEL, X)
    pdo = safe_predict_proba(PDO_MODEL, X)
    ga = GA_MODEL.predict(X)[0]

    # 防NaN
    pe = float(np.nan_to_num(pe))
    early = float(np.nan_to_num(early))
    pdo = float(np.nan_to_num(pdo))
    ga = float(np.nan_to_num(ga))

    # =========================
    # 输出
    # =========================
    st.divider()
    st.header("📊 预测结果")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("PE风险", f"{pe*100:.1f}%")
        st.write(risk_level(pe))

    with c2:
        st.metric("Early PE", f"{early*100:.1f}%")
        st.write(risk_level(early))

    with c3:
        st.metric("PDO风险", f"{pdo*100:.1f}%")
        st.write(risk_level(pdo))

    with c4:
        st.metric("分娩孕周", ga_format(ga))
        st.write(f"{ga:.2f} 周")

    st.divider()
    st.subheader("输入数据")
    st.dataframe(X)

    st.success("模型预测完成（仅科研用途）")
