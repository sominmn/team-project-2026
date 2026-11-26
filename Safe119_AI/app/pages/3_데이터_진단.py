
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="데이터 진단 | Safe119 AI",
    page_icon="🔍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"


@st.cache_data
def load_data():
    quality = pd.read_csv(
        DATA_DIR / "audit_year_quality.csv",
        encoding="utf-8-sig"
    )

    shift = pd.read_csv(
    DATA_DIR / "js_divergence_standardized.csv",
    encoding="utf-8-sig"
    )

    shift["비교연도"] = (
        shift["previous_year"].astype(str)
        + " → "
        + shift["current_year"].astype(str)
    )

    shift["JSD"] = shift["js_divergence"]

    summary = pd.read_csv(
        DATA_DIR / "quality_summary.csv",            encoding="utf-8-sig"
    )

    return quality, shift, summary


st.title("🔍 데이터 품질 및 분포 변화 진단")
st.caption(
    "119 구조활동 데이터의 품질과 연도별 사고원인 분포 변화를 분석합니다."
)

try:
    quality, shift, summary = load_data()
except Exception as e:
    st.error(f"데이터 로딩 오류: {e}")
    st.stop()

quality = quality.sort_values("연도").copy()

# ----------------------------
# 연도별 데이터 품질
# ----------------------------

st.markdown("### 📋 연도별 데이터 품질")

selected_year = st.selectbox(
    "진단할 연도",
    sorted(quality["연도"].unique().tolist()),
    index=len(quality) - 1
)

year_data = quality[
    quality["연도"] == selected_year
].iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "원본 데이터",
        f"{int(year_data['원본행수']):,}건"
    )

with col2:
    st.metric(
        "사고원인 결측",
        f"{int(year_data['사고원인결측']):,}건"
    )

with col3:
    st.metric(
        "날짜 파싱 실패",
        f"{int(year_data['날짜파싱실패']):,}건"
    )

with col4:
    st.metric(
        "사고원인 종류",
        f"{int(year_data['사고원인종류'])}개"
    )

st.caption(
    "※ 결측 건수와 날짜 파싱 실패 건수는 원본 데이터 진단 결과입니다."
)

st.divider()

# ----------------------------
# 연도별 원본 데이터 규모
# ----------------------------

st.markdown("### 📊 연도별 구조활동 데이터 규모")

fig_count = px.bar(
    quality,
    x="연도",
    y="원본행수",
    text="원본행수",
    title="연도별 원본 데이터 행 수"
)

fig_count.update_traces(
    texttemplate="%{text:,}",
    textposition="outside"
)

fig_count.update_layout(
    xaxis_title="연도",
    yaxis_title="원본 행 수"
)

fig_count.update_xaxes(type="category")

st.plotly_chart(
    fig_count,
    use_container_width=True
)

st.divider()

# ----------------------------
# 사고원인 종류 변화
# ----------------------------

st.markdown("### 🧩 사고원인 분류 변화")

fig_cause = px.line(
    quality,
    x="연도",
    y="사고원인종류",
    markers=True,
    title="연도별 사고원인 종류 수"
)

fig_cause.update_layout(
    xaxis_title="연도",
    yaxis_title="사고원인 종류 수"
)

fig_cause.update_xaxes(
    dtick=1
)

st.plotly_chart(
    fig_cause,
    use_container_width=True
)

st.info(
    "연도별 사고원인 종류 수의 차이는 분류체계나 "
    "데이터 구성의 변화를 점검할 필요가 있음을 보여줍니다. "
    "종류 수가 달라졌다는 사실만으로 공식 분류체계가 "
    "변경되었다고 단정할 수는 없습니다."
)

st.divider()

# ----------------------------
# JSD 분포 변화
# ----------------------------

st.markdown("### 📉 연도 간 사고원인 분포 변화")

fig_jsd = px.bar(
    shift,
    x="비교연도",
    y="JSD",
    text="JSD",
    title="연도 간 사고원인 분포 차이 (JSD)"
)

fig_jsd.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_jsd.update_layout(
    xaxis_title="비교 연도",
    yaxis_title="JSD"
)

st.plotly_chart(
    fig_jsd,
    use_container_width=True
)

st.caption(
    "JSD는 두 확률분포의 차이를 나타내는 지표입니다. "
    "값이 클수록 분포 차이가 크며, 계산 방식과 비교 대상에 "
    "따라 값이 달라질 수 있습니다."
)

st.divider()

# ----------------------------
# 2024-2025 사고원인 구성 변화
# ----------------------------

st.markdown("### 📊 2024→2025 사고원인 구성 변화")

@st.cache_data
def load_cause_counts():
    return pd.read_csv(
        DATA_DIR / "audit_cause_counts.csv",
        encoding="utf-8-sig"
    )

cause_df = load_cause_counts()

common_16 = cause_df[
    (cause_df["2024"] > 0)
    & (cause_df["2025"] > 0)
]

total_2024 = int(cause_df["2024"].sum())
total_2025 = int(cause_df["2025"].sum())

common_2024 = int(common_16["2024"].sum())
common_2025 = int(common_16["2025"].sum())

total_change_pct = (
    (total_2025 / total_2024 - 1) * 100
)

common_change_pct = (
    (common_2025 / common_2024 - 1) * 100
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 사고원인 기록 변화",
        f"{total_change_pct:+.2f}%",
        delta=f"{total_2025 - total_2024:,}건",
        delta_color="off"
    )

with col2:
    st.metric(
        "공통 16개 사고원인 기록 변화",
        f"{common_change_pct:+.2f}%",
        delta=f"{common_2025 - common_2024:,}건",
        delta_color="off"
    )

chart_df = pd.DataFrame({
    "연도": ["2024", "2025", "2024", "2025"],
    "비교대상": [
        "전체 사고원인",
        "전체 사고원인",
        "공통 16개 사고원인",
        "공통 16개 사고원인"
    ],
    "기록건수": [
        total_2024,
        total_2025,
        common_2024,
        common_2025
    ]
})

fig_change = px.bar(
    chart_df,
    x="연도",
    y="기록건수",
    color="비교대상",
    barmode="group",
    text="기록건수",
    title="전체 기록과 공통 사고원인 기록 비교"
)

fig_change.update_traces(
    texttemplate="%{text:,}",
    textposition="outside"
)

fig_change.update_layout(
    xaxis_title="연도",
    yaxis_title="기록 건수"
)

st.plotly_chart(
    fig_change,
    use_container_width=True
)

st.info(
    "2024년 대비 2025년 전체 사고원인 기록은 감소했지만, "
    "두 연도 모두 기록이 존재하는 공통 16개 사고원인 기록은 "
    "증가했습니다. 이는 비교 범위에 따라 변화 해석이 "
    "달라질 수 있음을 보여줍니다. "
    "단, 실제 구조 수요 변화나 공식 분류체계 변경의 "
    "원인을 확정하는 결과는 아닙니다."
)

st.divider()


# ----------------------------
# 지역별 월간 데이터 위험 신호
# ----------------------------

st.markdown("### 🚨 지역별 월간 데이터 위험 신호")

@st.cache_data
def load_risk_signals():
    return pd.read_csv(
        DATA_DIR / "risk_signals_monthly.csv",
        encoding="utf-8-sig"
    )

risk_df = load_risk_signals()

col_year, col_month, col_region = st.columns(3)

with col_year:
    risk_year = st.selectbox(
        "평가 연도",
        sorted(risk_df["연도"].unique().tolist()),
        index=len(risk_df["연도"].unique()) - 1,
        key="risk_year"
    )

with col_month:
    risk_month = st.selectbox(
        "평가 월",
        list(range(1, 13)),
        index=7,
        key="risk_month"
    )

with col_region:
    risk_region = st.selectbox(
        "평가 지역",
        sorted(risk_df["지역"].unique().tolist()),
        key="risk_region"
    )

selected_risk = risk_df[
    (risk_df["연도"] == risk_year)
    & (risk_df["월"] == risk_month)
    & (risk_df["지역"] == risk_region)
]

if selected_risk.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
else:
    risk = selected_risk.iloc[0]
    risk_level = risk["위험판정"]

    color_map = {
        "NORMAL": "green",
        "WATCH": "orange",
        "HIGH": "red",
        "INSUFFICIENT": "gray"
    }

    color = color_map.get(risk_level, "gray")

    st.markdown(
        f"**위험 판정:** :{color}[{risk_level}]"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "직전월 구조활동 건수",
            (
                f"{int(risk['직전월건수']):,}건"
                if pd.notna(risk["직전월건수"])
                else "데이터 없음"
            )
        )

    with col2:
        st.metric(
            "최근 변화율",
            (
                f"{risk['최근변화율'] * 100:.1f}%"
                if pd.notna(risk["최근변화율"])
                else "계산 불가"
            )
        )

    with col3:
        st.metric(
            "계절 변화율",
            (
                f"{risk['계절변화율'] * 100:.1f}%"
                if pd.notna(risk["계절변화율"])
                else "계산 불가"
            )
        )

    st.caption(
        "최근 변화율은 직전월과 전전월의 건수 차이, "
        "계절 변화율은 직전월과 전년도 같은 월의 "
        "건수 차이를 절댓값 비율로 계산합니다."
    )

    if risk_level == "INSUFFICIENT":
        st.info(
            "비교에 필요한 과거 데이터가 부족하여 "
            "위험 수준을 판정할 수 없습니다."
        )
    elif risk_level == "HIGH":
        st.error(
            "최근 변화와 계절 패턴 이탈이 동시에 "
            "감지되었습니다."
        )
    elif risk_level == "WATCH":
        st.warning(
            "최근 급변 또는 계절 패턴 이탈 중 "
            "하나가 감지되었습니다."
        )
    else:
        st.success(
            "설정된 기준을 초과하는 변화 신호가 "
            "감지되지 않았습니다."
        )

st.divider()

# ----------------------------
# 해당 월의 지역별 위험 판정 분포
# ----------------------------

st.markdown("### 📊 지역별 위험 판정 분포")

monthly_risk = risk_df[
    (risk_df["연도"] == risk_year)
    & (risk_df["월"] == risk_month)
]

risk_order = [
    "NORMAL", "WATCH", "HIGH", "INSUFFICIENT"
]

risk_counts = (
    monthly_risk["위험판정"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
    .reset_index()
)

risk_counts.columns = ["위험판정", "지역수"]

fig_risk = px.bar(
    risk_counts,
    x="위험판정",
    y="지역수",
    color="위험판정",
    text="지역수",
    title=f"{risk_year}년 {risk_month}월 지역별 위험 판정"
)

fig_risk.update_traces(
    textposition="outside"
)

fig_risk.update_layout(
    xaxis_title="위험 수준",
    yaxis_title="지역 수",
    showlegend=False
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

st.caption(
    "위험 신호는 예측 대상 월 이전에 관측된 "
    "구조활동 건수로 계산한 탐색적 지표입니다. "
    "실제 사고 발생 위험이나 예측 실패를 "
    "확정하는 지표는 아닙니다."
)

with st.expander("🔎 위험 신호 판정 기준"):
    st.markdown(
        """
        **최근 급변 신호**
        - 직전월과 전전월 건수의 절대 변화율이 30% 이상

        **계절 이탈 신호**
        - 직전월과 전년도 같은 월 건수의 절대 변화율이 40% 이상

        **위험 판정**
        - NORMAL: 감지된 신호 0개
        - WATCH: 감지된 신호 1개
        - HIGH: 감지된 신호 2개
        - INSUFFICIENT: 비교에 필요한 과거 데이터 부족

        ※ 30%, 40% 기준은 탐색적으로 설정한 값이며,
        최적화되거나 현장에서 검증된 기준은 아닙니다.
        """
    )

st.divider()

# ----------------------------
# 위험 신호별 예측 오차 분석
# ----------------------------

st.markdown("### 📈 위험 신호와 실제 예측 오차")

@st.cache_data
def load_risk_performance():
    performance = pd.read_csv(
        DATA_DIR / "risk_signal_performance.csv",
        encoding="utf-8-sig"
    )

    comparison = pd.read_csv(
        DATA_DIR / "risk_signal_comparison.csv",
        encoding="utf-8-sig"
    )

    return performance, comparison


performance_df, comparison_df = load_risk_performance()

analysis_year = st.selectbox(
    "예측 오차 분석 연도",
    sorted(performance_df["연도"].unique().tolist()),
    index=len(performance_df["연도"].unique()) - 1,
    key="risk_performance_year"
)

year_performance = performance_df[
    performance_df["연도"] == analysis_year
].copy()

risk_order = ["NORMAL", "WATCH", "HIGH"]

fig_performance = px.bar(
    year_performance,
    x="위험판정",
    y="MAE",
    color="모델",
    barmode="group",
    text="MAE",
    category_orders={"위험판정": risk_order},
    title=f"{analysis_year}년 위험 수준별 모델 예측 오차"
)

fig_performance.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)

fig_performance.update_layout(
    xaxis_title="위험 수준",
    yaxis_title="평균절대오차 MAE (건)",
    legend_title="모델"
)

st.plotly_chart(
    fig_performance,
    use_container_width=True
)

st.caption(
    "MAE는 실제 구조활동 건수와 예측 건수의 "
    "절대 차이를 평균한 값입니다. 낮을수록 정확합니다. "
    "위험 수준별 표본 수와 지역별 활동 규모가 다르므로 "
    "단순 비교만으로 인과관계를 판단할 수 없습니다."
)

st.markdown("#### 📊 위험 신호 유무에 따른 오차 비교")

year_comparison = comparison_df[
    comparison_df["연도"] == analysis_year
].copy()

fig_signal = px.bar(
    year_comparison,
    x="위험여부",
    y="MAE",
    color="모델",
    barmode="group",
    text="MAE",
    category_orders={
        "위험여부": ["NORMAL", "SIGNAL"]
    },
    title=f"{analysis_year}년 위험 신호 유무별 모델 성능"
)

fig_signal.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)

fig_signal.update_layout(
    xaxis_title="위험 신호",
    yaxis_title="평균절대오차 MAE (건)",
    legend_title="모델"
)

st.plotly_chart(
    fig_signal,
    use_container_width=True
)

st.info(
    "SIGNAL은 WATCH 또는 HIGH로 판정된 경우를 "
    "합친 그룹입니다. 이 그래프는 위험 신호와 "
    "예측 오차의 관계를 탐색하는 결과이며, "
    "위험 신호에 따라 예측 방식을 바꿨을 때 "
    "성능이 개선되는지를 직접 검증한 것은 아닙니다."
)

st.markdown("#### 📋 위험 수준별 상세 결과")

st.dataframe(
    year_performance[
        ["모델", "위험판정", "표본수", "MAE", "중앙절대오차"]
    ].round(2),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ----------------------------
# 위험 신호 기반 조치 효과 검증
# ----------------------------

st.markdown("### 🛠️ 위험 신호 기반 조치 효과 검증")

@st.cache_data
def load_switching_results():
    yearly = pd.read_csv(
        DATA_DIR / "model_switching_yearly.csv",
        encoding="utf-8-sig"
    )
    by_risk = pd.read_csv(
        DATA_DIR / "model_switching_by_risk.csv",
        encoding="utf-8-sig"
    )
    return yearly, by_risk


switch_yearly, switch_by_risk = load_switching_results()

st.caption(
    "위험 신호가 WATCH 또는 HIGH이면 이전 달 기준 예측을 사용하고, "
    "NORMAL이면 RF 잔차 고정형 0.25 예측을 사용하는 "
    "탐색적 전환 규칙을 평가합니다."
)

# 1. 연도별 전체 결과
chart_switch = switch_yearly.rename(columns={
    "year": "연도",
    "baseline_mae": "이전 달 기준",
    "always_ai_mae": "항상 AI 사용",
    "signal_switch_mae": "위험 신호 기반 전환"
})

chart_long = chart_switch.melt(
    id_vars="연도",
    value_vars=[
        "이전 달 기준",
        "항상 AI 사용",
        "위험 신호 기반 전환"
    ],
    var_name="예측 방식",
    value_name="MAE"
)

fig_switch = px.line(
    chart_long,
    x="연도",
    y="MAE",
    color="예측 방식",
    markers=True,
    title="연도별 예측 방식에 따른 MAE 비교"
)

fig_switch.update_layout(
    xaxis_title="평가 연도",
    yaxis_title="평균절대오차 MAE (건)"
)
fig_switch.update_xaxes(dtick=1)

st.plotly_chart(
    fig_switch,
    use_container_width=True
)

# 2. 연도별 조치 개선율
st.markdown("#### 📊 항상 AI 사용 대비 조치 개선율")

fig_improvement = px.bar(
    switch_yearly,
    x="year",
    y="switch_vs_ai_pct",
    text="switch_vs_ai_pct",
    title="위험 신호 기반 전환의 연도별 개선율"
)

fig_improvement.update_traces(
    texttemplate="%{text:+.2f}%",
    textposition="outside"
)
fig_improvement.update_layout(
    xaxis_title="평가 연도",
    yaxis_title="MAE 개선율 (%)"
)
fig_improvement.update_xaxes(type="category")
fig_improvement.add_hline(y=0, line_dash="dash")

st.plotly_chart(
    fig_improvement,
    use_container_width=True
)

# 3. 위험 수준별 상세 결과
st.markdown("#### 🔍 위험 수준별 조치 결과")

switch_detail_year = st.selectbox(
    "상세 결과 연도",
    sorted(switch_by_risk["year"].unique().tolist()),
    index=len(switch_by_risk["year"].unique()) - 1,
    key="switch_detail_year"
)

detail = switch_by_risk[
    switch_by_risk["year"] == switch_detail_year
].copy()

detail = detail.rename(columns={
    "risk_status": "위험판정",
    "count": "표본수",
    "baseline_mae": "이전 달 기준 MAE",
    "ai_mae": "항상 AI MAE",
    "switch_mae": "전환 MAE"
})

st.dataframe(
    detail[
        [
            "위험판정",
            "표본수",
            "이전 달 기준 MAE",
            "항상 AI MAE",
            "전환 MAE"
        ]
    ].round(2),
    use_container_width=True,
    hide_index=True
)

st.info(
    "2022~2024년에는 위험 신호 기반 전환이 항상 AI를 "
    "사용하는 방식보다 오차를 증가시켰고, "
    "2025년에만 오차가 감소했습니다. "
    "따라서 이 규칙의 효과가 모든 연도에서 "
    "일관되게 확인된 것은 아닙니다."
)

st.caption(
    "본 결과는 과거 데이터에 대한 탐색적 실험입니다. "
    "전환 규칙과 임계값의 독립적인 검증이 필요하며, "
    "실제 소방 인력·장비 배치에 바로 적용할 수 없습니다."
)

st.divider()



# ----------------------------
# 전체 vs 공통 9개 JSD 비교
# ----------------------------

st.markdown("### 🔎 전체 사고원인 vs 공통 9개 비교")

@st.cache_data
def load_common_jsd():
    return pd.read_csv(
        DATA_DIR / "common_vs_all_jsd.csv",
        encoding="utf-8-sig"
    )

common_jsd = load_common_jsd()

common_jsd["비교연도"] = (
    common_jsd["previous_year"].astype(str)
    + " → "
    + common_jsd["current_year"].astype(str)
)

comparison = common_jsd.melt(
    id_vars="비교연도",
    value_vars=["all_categories_jsd", "common_9_jsd"],
    var_name="비교대상",
    value_name="JS divergence"
)

comparison["비교대상"] = comparison["비교대상"].replace({
    "all_categories_jsd": "전체 사고원인",
    "common_9_jsd": "공통 9개 사고원인"
})

fig_comparison = px.bar(
    comparison,
    x="비교연도",
    y="JS divergence",
    color="비교대상",
    barmode="group",
    text="JS divergence",
    title="사고원인 비교 범위에 따른 분포 변화"
)

fig_comparison.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_comparison.update_layout(
    xaxis_title="비교 연도",
    yaxis_title="JS divergence",
    legend_title="비교 대상"
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)

st.info(
    "공통 9개 사고원인으로 비교 범위를 제한하면 "
    "연도별 분포 차이가 크게 줄어듭니다. "
    "다만 이는 공통 항목 내부의 상대적 구성비를 "
    "비교한 결과이며, 데이터 품질이나 예측 성능이 "
    "개선되었다는 의미는 아닙니다."
)

st.divider()



# ----------------------------
# 기존 진단 결과 요약
# ----------------------------

st.markdown("### 📝 데이터 진단 요약")

summary_display = summary.copy()
summary_display["값"] = pd.to_numeric(
    summary_display["값"],
    errors="coerce"
).round(4)

st.dataframe(
    summary_display,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.markdown("### 📂 연도별 상세 품질 진단")

st.dataframe(
    quality,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "📥 연도별 품질 진단 CSV 다운로드",
    data=quality.to_csv(index=False).encode("utf-8-sig"),
    file_name="safe119_year_quality.csv",
    mime="text/csv"
)

st.caption(
    "※ 본 페이지는 기존 분석 CSV를 시각화한 것으로, "
    "데이터 품질 문제의 원인이나 예측 성능에 미친 영향을 "
    "자동으로 판정하지 않습니다."
)

v3 = BASE_DIR / "reports/v3/signal_region_year.csv"
if v3.exists():
    st.subheader("시도·연도별 신호와 예측 오차")
    rows = pd.read_csv(v3)
    view_year = st.selectbox("진단 연도", sorted(rows.year.unique(), reverse=True))
    rows = rows.loc[rows.year.eq(view_year)].copy()
    rows["alert_rate"] *= 100
    st.scatter_chart(rows, x="alert_rate", y="excess_mae", x_label="경보율(%)", y_label="RF의 기준선 초과 MAE")
    st.dataframe(rows[["region", "months", "usable", "alerts", "alert_rate", "mae", "baseline_mae"]].rename(columns={
        "region": "시도", "months": "평가 월", "usable": "판정 가능 월", "alerts": "경보 월",
        "alert_rate": "경보율(%)", "mae": "RF MAE", "baseline_mae": "전월 기준 MAE",
    }), hide_index=True)
    st.caption("후향 평가입니다. 지역별 결과는 전국 공통 분류 변화의 영향을 공유할 수 있습니다.")
