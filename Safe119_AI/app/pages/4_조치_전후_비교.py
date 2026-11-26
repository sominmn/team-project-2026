"""Compare retrospective predictions and signal-triggered actions."""
from pathlib import Path

import pandas as pd
import streamlit as st

REPORTS = Path(__file__).resolve().parents[2] / "reports/v3"
ACTIONS = {"전월 기준선": "baseline", "최근 2년만 재학습": "recent_rf", "변화율 예측": "growth_rf"}
POLICIES = {"none": "조치 없음", "always": "항상 조치", "signal_only": "신호 시 조치"}
st.set_page_config(page_title="조치 전후 예측 비교", layout="wide")
st.title("조치 전후 예측 비교")
st.caption("2022~2025년 후향 평가 · 2025년은 탐색 사례")
if not (REPORTS / "predictions.csv").exists():
    st.info("프로젝트 폴더에서 python evaluate_v3.py를 실행하면 비교 결과가 생성됩니다.")
    st.stop()


@st.cache_data
def load(name):
    return pd.read_csv(REPORTS / f"{name}.csv")


predictions = load("predictions").query("target == 'all' and not weather")
year = st.selectbox("평가 연도", sorted(predictions.year.unique(), reverse=True))
region = st.selectbox("시도", sorted(predictions.region.unique()))
action = ACTIONS[st.selectbox("조치", list(ACTIONS))]
sample = predictions.loc[predictions.year.eq(year) & predictions.region.eq(region)].set_index("month")
chart = pd.DataFrame({
    "실제 건수": sample.actual,
    "조치 없음": sample.rf,
    "항상 조치": sample[action],
    "신호 시 조치": sample[action].where(sample.signal.eq(True), sample.rf),
})
st.line_chart(chart)
st.subheader("전국 평가")
scores = load("action_scores").query("year == @year and action == @action").copy()
scores.policy = scores.policy.map(POLICIES)
st.dataframe(scores[["policy", "n", "mae", "baseline_mae", "improvement_pct"]].rename(columns={
    "policy": "전략", "n": "평가 행", "mae": "MAE", "baseline_mae": "기준선 MAE",
    "improvement_pct": "기준선 대비 개선율(%)",
}), hide_index=True)
st.subheader("해당 시도의 위험 신호와 조치 효과")
alerts = load("alert_detail").query("year == @year and region == @region and action == @action").copy()
alerts.outcome = alerts.outcome.map({
    "unknown": "판정 유보", "helpful_alert": "유효 경보", "false_alarm": "조치 효과 없는 경보",
    "missed_benefit": "조치가 유리했으나 경보 없음", "no_alert_no_benefit": "경보 및 조치 효과 없음",
})
st.dataframe(alerts[["month", "rf_error", "action_error", "outcome"]].rename(columns={
    "month": "월", "rf_error": "RF 절대오차", "action_error": "조치 절대오차", "outcome": "판정",
}), hide_index=True)
