
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="AI 수요 예측 | Safe119",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"


@st.cache_data
def load_predictions():
    # 기존 모델들의 예측 결과
    advanced = pd.read_csv(
        DATA_DIR / "advanced_model_predictions.csv",
        encoding="utf-8-sig"
    )

    advanced = advanced.rename(
        columns={"평가연도": "연도"}
    )

    advanced = advanced[
        ["연도", "월", "지역", "모델", "실제건수", "예측건수"]
    ]

    # 적응형 잔차 AI 모델의 예측 결과
    adaptive = pd.read_csv(
        DATA_DIR / "adaptive_residual_predictions.csv",
        encoding="utf-8-sig"
    )

    adaptive = adaptive.rename(
        columns={
            "구조활동건수": "실제건수",
            "적응형예측": "예측건수"
        }
    )

    adaptive["모델"] = "적응형 잔차 AI"

    adaptive = adaptive[
        ["연도", "월", "지역", "모델", "실제건수", "예측건수"]
    ]

    # 두 데이터 합치기
    df = pd.concat(
        [advanced, adaptive],
        ignore_index=True
    )

    df["연도"] = df["연도"].astype(int)
    df["월"] = df["월"].astype(int)

    df["실제건수"] = pd.to_numeric(
        df["실제건수"], errors="coerce"
    )
    df["예측건수"] = pd.to_numeric(
        df["예측건수"], errors="coerce"
    )

    df = df.dropna(
        subset=["실제건수", "예측건수"]
    )

    # 예측 오차
    df["절대오차"] = (
        df["실제건수"] - df["예측건수"]
    ).abs()

    df["제곱오차"] = (
        df["실제건수"] - df["예측건수"]
    ) ** 2

    return df


st.title("🤖 AI 구조활동 수요 예측")
st.caption(
    "과거 데이터에 대한 AI 모델의 예측 결과와 실제 구조활동 건수를 비교합니다."
)

try:
    df = load_predictions()
except Exception as e:
    st.error(f"예측 데이터 로딩 오류: {e}")
    st.stop()

# -------------------------
# 분석 조건
# -------------------------

st.sidebar.header("🤖 AI 예측 분석 조건")

years = sorted(df["연도"].unique().tolist())
regions = sorted(df["지역"].unique().tolist())
models = sorted(df["모델"].unique().tolist())

selected_year = st.sidebar.selectbox(
    "평가 연도",
    years,
    index=len(years) - 1
)

selected_region = st.sidebar.selectbox(
    "분석 지역",
    ["전국"] + regions
)

selected_models = st.sidebar.multiselect(
    "비교할 AI 모델",
    models,
    default=models
)

filtered = df[df["연도"] == selected_year].copy()

if selected_region != "전국":
    filtered = filtered[
        filtered["지역"] == selected_region
    ]

filtered = filtered[
    filtered["모델"].isin(selected_models)
]

if filtered.empty:
    st.warning("선택한 조건에 해당하는 예측 결과가 없습니다.")
    st.stop()

st.markdown(
    f"### 📍 {selected_year}년 {selected_region} AI 예측 결과"
)

# -------------------------
# 모델 성능 비교
# -------------------------

import numpy as np

performance = (
    filtered.groupby("모델")
    .agg(
        MAE=("절대오차", "mean"),
        MSE=("제곱오차", "mean"),
        평가건수=("절대오차", "count")
    )
    .reset_index()
)

performance["RMSE"] = np.sqrt(
    performance["MSE"]
)

performance = performance.sort_values("MAE")

best_model = performance.iloc[0]["모델"]
best_mae = performance.iloc[0]["MAE"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최저 MAE 모델",
        best_model
    )

with col2:
    st.metric(
        "최저 MAE",
        f"{best_mae:,.2f}건"
    )

with col3:
    st.metric(
        "비교 모델 수",
        f"{len(performance)}개"
    )

st.divider()

st.markdown("### 📊 모델별 예측 오차 비교")

fig_mae = px.bar(
    performance,
    x="모델",
    y="MAE",
    color="모델",
    text="MAE",
    title="모델별 평균 절대 오차 (MAE)"
)

fig_mae.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)

fig_mae.update_layout(
    showlegend=False,
    xaxis_title="AI 모델",
    yaxis_title="MAE (건)"
)

st.plotly_chart(
    fig_mae,
    use_container_width=True
)

st.dataframe(
    performance[
        ["모델", "MAE", "RMSE", "평가건수"]
    ].round(2),
    use_container_width=True,
    hide_index=True
)

st.divider()

# -------------------------
# 실제값 vs 예측값
# -------------------------

st.markdown("### 📈 실제 구조활동 vs AI 예측")

graph_models = st.multiselect(
    "그래프에 표시할 모델",
    performance["모델"].tolist(),
    default=performance["모델"].tolist()[:2]
)

graph_df = filtered[
    filtered["모델"].isin(graph_models)
].copy()

if not graph_df.empty:
    # 전국 선택 시 지역별 예측값을 합산
    monthly = (
        graph_df.groupby(
            ["월", "모델"],
            as_index=False
        )[["실제건수", "예측건수"]]
        .sum()
    )

    # 실제값은 모델별로 중복 표시하지 않음
    actual = (
        monthly.groupby("월", as_index=False)
        ["실제건수"]
        .first()
    )

    actual["구분"] = "실제 구조활동"
    actual = actual.rename(
        columns={"실제건수": "건수"}
    )

    predicted = monthly.rename(
        columns={"예측건수": "건수"}
    )
    predicted["구분"] = predicted["모델"]

    plot_df = pd.concat(
        [
            actual[["월", "건수", "구분"]],
            predicted[["월", "건수", "구분"]]
        ],
        ignore_index=True
    )

    fig = px.line(
        plot_df,
        x="월",
        y="건수",
        color="구분",
        markers=True,
        title="월별 실제값과 AI 예측값 비교"
    )

    fig.update_layout(
        xaxis_title="월",
        yaxis_title="구조활동 건수",
        legend_title="구분"
    )

    fig.update_xaxes(
        dtick=1
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

st.markdown("### 📋 예측 결과 상세 데이터")

st.dataframe(
    filtered[
        [
            "연도",
            "월",
            "지역",
            "모델",
            "실제건수",
            "예측건수",
            "절대오차"
        ]
    ].round(2),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "※ 본 화면의 예측값은 과거 데이터에 대한 모델 평가 결과이며, "
    "실시간 또는 미래 구조활동 예측값이 아닙니다."
)
