
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"

risk = pd.read_csv(
    DATA / "risk_signals_monthly.csv",
    encoding="utf-8-sig"
)

pred = pd.read_csv(
    DATA / "advanced_model_predictions.csv",
    encoding="utf-8-sig"
)

# Use only two models for the initial comparison.
models = ["이전 달 기준", "RF 잔차 고정형 0.25"]
pred = pred[pred["모델"].isin(models)].copy()

# Align the prediction year column with risk signal data.
pred = pred.rename(columns={"평가연도": "연도"})

keys = ["연도", "월", "지역"]

if risk.duplicated(keys).any():
    raise ValueError("Duplicate risk signal keys detected.")

if pred.duplicated(keys + ["모델"]).any():
    raise ValueError("Duplicate prediction keys detected.")

merged = pred.merge(
    risk[keys + ["위험판정", "최근변화율", "계절변화율"]],
    on=keys,
    how="left",
    validate="many_to_one",
    indicator=True
)

print("===== MERGE CHECK =====")
print(merged["_merge"].value_counts().to_string())

if (merged["_merge"] != "both").any():
    raise ValueError("Some predictions have no risk signal.")

merged = merged.drop(columns="_merge")

merged["절대오차"] = (
    merged["실제건수"] - merged["예측건수"]
).abs()

# Exclude cases without enough historical data.
valid = merged[
    merged["위험판정"].isin(["NORMAL", "WATCH", "HIGH"])
].copy()

summary = (
    valid.groupby(["연도", "모델", "위험판정"])
    .agg(
        표본수=("절대오차", "size"),
        MAE=("절대오차", "mean"),
        중앙절대오차=("절대오차", "median")
    )
    .reset_index()
)

summary["MAE"] = summary["MAE"].round(2)
summary["중앙절대오차"] = summary["중앙절대오차"].round(2)

print("\n===== RISK GROUP PERFORMANCE =====")
print(summary.to_string(index=False))

# Compare any detected risk against normal cases.
valid["위험여부"] = np.where(
    valid["위험판정"] == "NORMAL",
    "NORMAL",
    "SIGNAL"
)

comparison = (
    valid.groupby(["연도", "모델", "위험여부"])
    .agg(
        표본수=("절대오차", "size"),
        MAE=("절대오차", "mean")
    )
    .reset_index()
)

print("\n===== NORMAL VS SIGNAL =====")
print(comparison.round(2).to_string(index=False))

summary.to_csv(
    DATA / "risk_signal_performance.csv",
    index=False,
    encoding="utf-8-sig"
)

comparison.to_csv(
    DATA / "risk_signal_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print("risk_signal_performance.csv")
print("risk_signal_comparison.csv")
