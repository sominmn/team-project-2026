
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"

df = pd.read_csv(
    DATA / "advanced_model_predictions.csv",
    encoding="utf-8-sig"
)

df["절대오차"] = abs(
    df["실제건수"] - df["예측건수"]
)
df["제곱오차"] = (
    df["실제건수"] - df["예측건수"]
) ** 2
df["오차"] = df["예측건수"] - df["실제건수"]

# 지역별·연도별·모델별 평가
results = (
    df.groupby(["평가연도", "지역", "모델"])
    .agg(
        MAE=("절대오차", "mean"),
        MSE=("제곱오차", "mean"),
        실제합계=("실제건수", "sum"),
        절대오차합계=("절대오차", "sum"),
        평균편향=("오차", "mean"),
        최대절대오차=("절대오차", "max")
    )
    .reset_index()
)

results["RMSE"] = np.sqrt(results["MSE"])
results["WAPE"] = np.where(
    results["실제합계"] > 0,
    results["절대오차합계"]
    / results["실제합계"] * 100,
    np.nan
)

results = results.drop(
    columns=["MSE", "절대오차합계"]
)

results.to_csv(
    DATA / "regional_model_performance.csv",
    index=False,
    encoding="utf-8-sig"
)

# 2022~2024년 지역별 모델 비교
selection = results[
    results["평가연도"].between(2022, 2024)
]

regional = (
    selection.groupby(["지역", "모델"])
    .agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        실제합계=("실제합계", "sum")
    )
    .reset_index()
)

# 적응형 RF와 이전 달 기준 비교
pivot = regional.pivot(
    index="지역",
    columns="모델",
    values="MAE"
)

comparison = pd.DataFrame({
    "지역": pivot.index,
    "기준_MAE": pivot["이전 달 기준"],
    "적응형_MAE": pivot["RF 잔차 적응형"]
})

comparison["개선율(%)"] = (
    (comparison["기준_MAE"] - comparison["적응형_MAE"])
    / comparison["기준_MAE"] * 100
)

comparison = comparison.sort_values(
    "개선율(%)",
    ascending=False
)

comparison.to_csv(
    DATA / "regional_improvement.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 2022~2024 지역별 AI 개선율 =====")
print(comparison.round(2).to_string(index=False))

improved = (comparison["개선율(%)"] > 0).sum()

print("\n===== 지역별 개선 요약 =====")
print(f"AI 성능 개선 지역: {improved}/17")
print(
    f"평균 지역 개선율: "
    f"{comparison['개선율(%)'].mean():.2f}%"
)
print(
    f"가장 높은 개선율: "
    f"{comparison.iloc[0]['지역']}"
)
print(
    f"가장 낮은 개선율: "
    f"{comparison.iloc[-1]['지역']}"
)

# 2025년 지역별 결과도 별도 확인
recent = results[
    results["평가연도"] == 2025
]

recent_pivot = recent.pivot(
    index="지역",
    columns="모델",
    values="MAE"
)

recent_compare = pd.DataFrame({
    "지역": recent_pivot.index,
    "기준_MAE": recent_pivot["이전 달 기준"],
    "적응형_MAE": recent_pivot["RF 잔차 적응형"]
})

recent_compare["개선율(%)"] = (
    (recent_compare["기준_MAE"]
     - recent_compare["적응형_MAE"])
    / recent_compare["기준_MAE"] * 100
)

print("\n===== 2025 지역별 개선율 =====")
print(
    recent_compare.sort_values(
        "개선율(%)",
        ascending=False
    ).round(2).to_string(index=False)
)

print("\n지역별 성능 분석 완료!")
