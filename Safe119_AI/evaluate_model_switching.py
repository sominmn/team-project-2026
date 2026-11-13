
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "processed"

pred = pd.read_csv(
    DATA / "advanced_model_predictions.csv",
    encoding="utf-8-sig"
)
risk = pd.read_csv(
    DATA / "risk_signals_monthly.csv",
    encoding="utf-8-sig"
)

# Avoid Korean literals in the model-selection logic.
cols = pred.columns.tolist()
pred.columns = [
    "year", "month", "region",
    "model", "actual", "predicted"
]

risk = risk.rename(columns={
    risk.columns[0]: "year",
    risk.columns[1]: "month",
    risk.columns[2]: "region",
    risk.columns[-1]: "risk_status"
})

models = pred["model"].drop_duplicates().tolist()
print("Models:", models)

baseline_name = models[0]
ai_name = models[4]

selected = pred[
    pred["model"].isin([baseline_name, ai_name])
].copy()

selected["model_type"] = np.where(
    selected["model"] == baseline_name,
    "BASELINE",
    "AI"
)

wide = selected.pivot(
    index=["year", "month", "region", "actual"],
    columns="model_type",
    values="predicted"
).reset_index()

merged = wide.merge(
    risk[["year", "month", "region", "risk_status"]],
    on=["year", "month", "region"],
    how="left",
    validate="one_to_one"
)

if merged[["BASELINE", "AI", "risk_status"]].isna().any().any():
    raise ValueError("Missing prediction or risk data.")

merged = merged[
    merged["risk_status"] != "INSUFFICIENT"
].copy()

merged["baseline_error"] = (
    merged["actual"] - merged["BASELINE"]
).abs()

merged["ai_error"] = (
    merged["actual"] - merged["AI"]
).abs()

# Experimental rule: switch to baseline on WATCH or HIGH.
merged["switch_prediction"] = np.where(
    merged["risk_status"].isin(["WATCH", "HIGH"]),
    merged["BASELINE"],
    merged["AI"]
)

merged["switch_error"] = (
    merged["actual"] - merged["switch_prediction"]
).abs()

rows = []

for (year, status), group in merged.groupby(
    ["year", "risk_status"]
):
    base_mae = group["baseline_error"].mean()
    ai_mae = group["ai_error"].mean()
    switch_mae = group["switch_error"].mean()

    rows.append({
        "year": year,
        "risk_status": status,
        "count": len(group),
        "baseline_mae": round(base_mae, 2),
        "ai_mae": round(ai_mae, 2),
        "switch_mae": round(switch_mae, 2),
        "ai_improvement_pct": round(
            (base_mae - ai_mae) / base_mae * 100, 2
        ) if base_mae != 0 else np.nan
    })

summary = pd.DataFrame(rows)

print("\n===== PERFORMANCE BY RISK =====")
print(summary.to_string(index=False))

print("\n===== YEARLY STRATEGY COMPARISON =====")

yearly = (
    merged.groupby("year")
    .agg(
        count=("actual", "size"),
        baseline_mae=("baseline_error", "mean"),
        always_ai_mae=("ai_error", "mean"),
        signal_switch_mae=("switch_error", "mean")
    )
    .round(2)
)

yearly["switch_vs_ai_pct"] = (
    (yearly["always_ai_mae"] - yearly["signal_switch_mae"])
    / yearly["always_ai_mae"] * 100
).round(2)

print(yearly.to_string())

summary.to_csv(
    DATA / "model_switching_by_risk.csv",
    index=False,
    encoding="utf-8-sig"
)

yearly.to_csv(
    DATA / "model_switching_yearly.csv",
    encoding="utf-8-sig"
)

print("\nSaved model switching results.")
