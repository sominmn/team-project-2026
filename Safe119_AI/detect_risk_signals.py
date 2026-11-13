
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"

source = DATA / "monthly_rescue_counts.csv"
df = pd.read_csv(source, encoding="utf-8-sig")

required = ["연도", "월", "지역", "구조활동건수"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Convert year and month to a monthly date.
df["날짜"] = pd.to_datetime(
    dict(year=df["연도"], month=df["월"], day=1)
)

df = df.sort_values(["지역", "날짜"]).copy()

if df.duplicated(["지역", "날짜"]).any():
    raise ValueError("Duplicate region-month rows detected.")

# Ensure missing months are not silently treated as adjacent.
parts = []
for region, group in df.groupby("지역"):
    group = group.set_index("날짜").sort_index()
    full_index = pd.date_range(
        group.index.min(), group.index.max(), freq="MS"
    )
    group = group.reindex(full_index)
    group["지역"] = region
    parts.append(group.rename_axis("날짜").reset_index())

df = pd.concat(parts, ignore_index=True)
df = df.sort_values(["지역", "날짜"])

# Historical values only: the target month's count is not used.
grouped = df.groupby("지역")["구조활동건수"]
df["직전월건수"] = grouped.shift(1)
df["전전월건수"] = grouped.shift(2)
df["전년직전월건수"] = grouped.shift(13)

df["최근변화율"] = (
    (df["직전월건수"] - df["전전월건수"]).abs()
    / df["전전월건수"].replace(0, np.nan)
)

df["계절변화율"] = (
    (df["직전월건수"] - df["전년직전월건수"]).abs()
    / df["전년직전월건수"].replace(0, np.nan)
)

# Exploratory thresholds only; not yet validated.
RECENT_THRESHOLD = 0.30
SEASONAL_THRESHOLD = 0.40

df["최근급변신호"] = np.where(
    df["최근변화율"].notna(),
    df["최근변화율"] >= RECENT_THRESHOLD,
    pd.NA,
)

df["계절이탈신호"] = np.where(
    df["계절변화율"].notna(),
    df["계절변화율"] >= SEASONAL_THRESHOLD,
    pd.NA,
)

df["위험신호개수"] = (
    df[["최근급변신호", "계절이탈신호"]]
    .fillna(False)
    .astype(int)
    .sum(axis=1)
)

df["위험판정"] = np.select(
    [
        df["위험신호개수"] == 2,
        df["위험신호개수"] == 1,
    ],
    ["HIGH", "WATCH"],
    default="NORMAL",
)

# Insufficient historical data must not be labeled NORMAL.
insufficient = (
    df["최근변화율"].isna()
    | df["계절변화율"].isna()
)
df.loc[insufficient, "위험판정"] = "INSUFFICIENT"

df["연도"] = df["날짜"].dt.year
df["월"] = df["날짜"].dt.month

columns = [
    "연도", "월", "지역",
    "직전월건수", "전전월건수", "전년직전월건수",
    "최근변화율", "계절변화율",
    "최근급변신호", "계절이탈신호",
    "위험신호개수", "위험판정",
]

result = df[columns].copy()

output = DATA / "risk_signals_monthly.csv"
result.to_csv(output, index=False, encoding="utf-8-sig")

print("\n===== RISK SIGNAL SUMMARY =====")
print("Rows:", len(result))
print(result["위험판정"].value_counts().to_string())

print("\n===== 2025 SAMPLE =====")
print(
    result[result["연도"] == 2025]
    .head(10)
    .to_string(index=False)
)

print("\nSaved:", output)
