
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

# AI 예측 결과
pred = pd.read_csv(
    DATA_DIR / "final_predictions.csv",
    encoding="utf-8-sig"
)

pred["날짜"] = pd.to_datetime(pred["날짜"])
pred["월"] = pred["날짜"].dt.month

# 월별 실제 건수와 예측 건수 비교
monthly = pred.groupby("월").agg(
    실제건수=("구조활동건수", "sum"),
    AI예측건수=("예측건수", "sum")
).reset_index()

monthly["예측차이"] = (
    monthly["AI예측건수"] - monthly["실제건수"]
)

monthly["절대차이"] = monthly["예측차이"].abs()

print("\n===== 2025년 월별 예측 비교 =====")
print(monthly.round(1).to_string(index=False))

# 지역별 오차
pred["절대오차"] = (
    pred["예측건수"] - pred["구조활동건수"]
).abs()

regional = (
    pred.groupby("지역")["절대오차"]
    .mean()
    .sort_values(ascending=False)
)

print("\n===== 지역별 평균 절대오차 =====")
print(regional.round(1).to_string())

# 가장 크게 틀린 예측 10개
print("\n===== 오차가 큰 예측 TOP 10 =====")

top_errors = pred.nlargest(
    10, "절대오차"
)

print(
    top_errors[
        ["날짜", "지역", "구조활동건수",
         "예측건수", "절대오차"]
    ].round(1).to_string(index=False)
)

monthly.to_csv(
    DATA_DIR / "monthly_prediction_errors.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n분석 완료!")
