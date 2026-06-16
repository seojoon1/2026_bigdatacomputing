import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────────────────────
# 0. 공통 설정 (notebook의 학습 코드와 반드시 동일하게 유지!)
# ─────────────────────────────────────────────────────────────
FEATURES = ["Adult mortality", "BMI", "GDP"]   # 자율 선택 3개 (Schooling 금지)
TARGET = "Life expectancy"
DATA_URL = "https://github.com/dongupak/DataML/raw/main/csv/life_expectancy.csv"
RANDOM_STATE = 42
TRAIN_SAMPLE_N = 50

st.set_page_config(page_title="기대수명 예측 서비스", layout="wide")


# ─────────────────────────────────────────────────────────────
# 1. 데이터 로드 + 학습/테스트 분할 (notebook과 동일 seed로 재현)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_split():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip()      # WHO 데이터셋 컬럼명 공백 제거
    df = df.dropna()
    X, y = df[FEATURES], df[TARGET]
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    # 과적합 관찰용: 훈련은 무작위 50개만
    X_tr = X_tr_full.sample(n=TRAIN_SAMPLE_N, random_state=RANDOM_STATE)
    y_tr = y_tr_full.loc[X_tr.index]
    return df, X_tr, X_te, y_tr, y_te


@st.cache_resource
def load_models():
    return {name: joblib.load(f"model_{name}.pkl") for name in ["Linear", "Poly", "Ridge"]}


df, X_train, X_test, y_train, y_test = load_split()
models = load_models()


# ─────────────────────────────────────────────────────────────
# 2. 성능 평가지표 테이블
# ─────────────────────────────────────────────────────────────
def build_metrics():
    rows = []
    for name, m in models.items():
        pred_tr, pred_te = m.predict(X_train), m.predict(X_test)
        rows.append({
            "Model": name,
            "Train R²": round(r2_score(y_train, pred_tr), 3),
            "Test R²": round(r2_score(y_test, pred_te), 3),
            "Train MSE": round(mean_squared_error(y_train, pred_tr), 2),
            "Test MSE": round(mean_squared_error(y_test, pred_te), 2),
            "Complexity(특성수)": m.named_steps["poly"].n_output_features_,
        })
    return pd.DataFrame(rows).set_index("Model")


metrics_df = build_metrics()

# ─────────────────────────────────────────────────────────────
# 화면 구성
# ─────────────────────────────────────────────────────────────
st.title("🩺 다중 특성 회귀 모델 — 기대수명 예측")
st.caption(f"독립변수: {', '.join(FEATURES)} | 훈련 샘플 {TRAIN_SAMPLE_N}개 (과적합 유도)")

st.header("1️⃣ 모델 성능 비교")
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("성능 평가지표")
    st.dataframe(metrics_df, use_container_width=True)
    st.info("Poly(3차·규제X)는 Train R²↑ Test R²↓ → 과적합. Ridge가 규제로 일반화 성능을 잡아줍니다.")

with col2:
    st.subheader("Test R² 비교")
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax.bar(metrics_df.index, metrics_df["Test R²"], color=colors)
    ax.set_ylabel("Test R²")
    ax.set_title("Model Generalization (Test R²)")
    ax.axhline(0, color="gray", lw=0.8)
    for i, v in enumerate(metrics_df["Test R²"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top")
    st.pyplot(fig)

st.divider()

# ─────────────────────────────────────────────────────────────
# 3. 실시간 예측 UI
# ─────────────────────────────────────────────────────────────
st.header("2️⃣ 실시간 기대수명 예측")

st.sidebar.header("⚙️ 입력값 조절")
user_input = {}
for feat in FEATURES:
    lo, hi = float(df[feat].min()), float(df[feat].max())
    mean = float(df[feat].mean())
    user_input[feat] = st.sidebar.slider(feat, lo, hi, mean)

choice = st.sidebar.selectbox("예측 모델 선택", ["Linear", "Poly", "Ridge"])

input_df = pd.DataFrame([user_input])[FEATURES]
pred = models[choice].predict(input_df)[0]

st.markdown(f"### 선택 모델: `{choice}`")
st.markdown(
    f"<h1 style='text-align:center;color:#2E86C1;'>예측 기대수명: {pred:.1f} 세</h1>",
    unsafe_allow_html=True,
)
st.write("입력값:", user_input)
