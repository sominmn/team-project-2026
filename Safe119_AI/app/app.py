
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Safe119 AI",
    page_icon="🚒",
    layout="wide"
)

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "monthly_rescue_counts.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    df["연도"] = pd.to_numeric(df["연도"], errors="raise")
    df["월"] = pd.to_numeric(df["월"], errors="raise")
    df["구조활동건수"] = pd.to_numeric(
        df["구조활동건수"], errors="raise"
    )

    return df


st.title("🚒 Safe119 AI")
st.subheader("AI 기반 119 구조활동 수요 예측 및 분석 플랫폼")

st.caption("소방청 구조활동 공공데이터 기반 | 2020~2025년")

st.divider()

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# 사이드바
st.sidebar.header("📊 분석 조건")

years = sorted(df["연도"].unique().tolist())
regions = sorted(df["지역"].unique().tolist())

selected_year = st.sidebar.selectbox(
    "분석 연도",
    years,
    index=len(years) - 1
)

selected_region = st.sidebar.selectbox(
    "분석 지역",
    ["전국"] + regions
)

# 선택한 연도의 데이터
filtered = df[df["연도"] == selected_year].copy()

if selected_region != "전국":
    filtered = filtered[
        filtered["지역"] == selected_region
    ]

# 월별 합계 (전국 선택 시 모든 지역 합산)
monthly = (
    filtered.groupby("월", as_index=False)["구조활동건수"]
    .sum()
    .sort_values("월")
)

if monthly.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# 주요 지표
total_count = int(monthly["구조활동건수"].sum())
monthly_avg = monthly["구조활동건수"].mean()

peak_row = monthly.loc[
    monthly["구조활동건수"].idxmax()
]
peak_month = int(peak_row["월"])

st.markdown(
    f"### 📍 {selected_year}년 {selected_region} 구조활동 현황"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "연간 구조활동",
        f"{total_count:,}건"
    )

with col2:
    st.metric(
        "월평균 구조활동",
        f"{monthly_avg:,.0f}건"
    )

with col3:
    st.metric(
        "최다 구조활동 월",
        f"{peak_month}월"
    )

st.divider()

# 월별 추이 그래프
st.markdown("### 📈 월별 구조활동 추이")

chart_df = monthly.copy()
chart_df["월"] = chart_df["월"].astype(str) + "월"
chart_df = chart_df.set_index("월")

st.line_chart(
    chart_df["구조활동건수"],
    use_container_width=True
)

# 데이터 표
st.markdown("### 📋 월별 상세 데이터")

display_df = monthly.rename(
    columns={
        "월": "월",
        "구조활동건수": "구조활동 건수"
    }
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "※ 본 화면은 과거 구조활동 통계 분석이며, "
    "AI 예측 결과는 추후 별도 화면에서 제공됩니다."
)
