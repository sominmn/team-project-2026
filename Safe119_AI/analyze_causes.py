
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

results = []

for year in [2024, 2025]:
    file = RAW_DIR / f"소방청_구조활동현황_{year}1231.csv"

    print(f"\n{year}년 원본 분석 중...")

    for chunk in pd.read_csv(
        file,
        encoding="cp949",
        usecols=["신고년월일", "사고원인"],
        chunksize=50000
    ):
        chunk["신고년월일"] = pd.to_datetime(
            chunk["신고년월일"],
            errors="coerce"
        )

        chunk = chunk.dropna(
            subset=["신고년월일"]
        ).copy()

        chunk["월"] = chunk["신고년월일"].dt.month

        # 여름철 7~9월만 비교
        chunk = chunk[
            chunk["월"].between(7, 9)
        ].copy()

        chunk["사고원인"] = (
            chunk["사고원인"]
            .astype("string")
            .str.strip()
            .fillna("미상")
        )

        counts = (
            chunk.groupby("사고원인")
            .size()
            .reset_index(name="건수")
        )

        counts["연도"] = year
        results.append(counts)

df = pd.concat(results, ignore_index=True)

summary = (
    df.groupby(["연도", "사고원인"])["건수"]
    .sum()
    .reset_index()
)

pivot = summary.pivot_table(
    index="사고원인",
    columns="연도",
    values="건수",
    fill_value=0
)

for year in [2024, 2025]:
    if year not in pivot.columns:
        pivot[year] = 0

pivot["증감"] = pivot[2025] - pivot[2024]

pivot["증감률(%)"] = (
    pivot["증감"]
    .div(pivot[2024].replace(0, float("nan")))
    .mul(100)
    .round(1)
)

pivot = pivot.sort_values("증감")

print("\n===== 2024 vs 2025 여름철 사고원인 변화 =====")
print(pivot.head(20).to_string())

output_path = (
    OUTPUT_DIR / "summer_cause_comparison.csv"
)

pivot.to_csv(
    output_path,
    encoding="utf-8-sig"
)

print("\n저장 위치:", output_path)
print("\n분석 완료!")
