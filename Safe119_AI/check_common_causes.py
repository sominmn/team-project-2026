
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

yearly_counts = {}

for year in range(2020, 2026):
    file_path = (
        RAW_DIR / f"소방청_구조활동현황_{year}1231.csv"
    )

    print(f"{year}년 분석 중...")

    counts = pd.Series(dtype="int64")

    for chunk in pd.read_csv(
        file_path,
        encoding="cp949",
        usecols=["사고원인"],
        chunksize=50000
    ):
        causes = (
            chunk["사고원인"]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .fillna("미상")
        )

        counts = counts.add(
            causes.value_counts(),
            fill_value=0
        )

    yearly_counts[year] = counts.astype("int64")

# 모든 연도의 사고원인별 건수 표
comparison = pd.DataFrame(yearly_counts).fillna(0)
comparison = comparison.astype("int64")

# 6개 연도 모두에서 1건 이상 등장한 사고원인
common_mask = (comparison > 0).all(axis=1)

common_causes = comparison[common_mask].copy()

# 미상은 공통 분류에서 제외
common_causes = common_causes.drop(
    index="미상",
    errors="ignore"
)

print("\n===== 연도별 사고원인 종류 수 =====")

for year in range(2020, 2026):
    print(
        f"{year}년: "
        f"{(comparison[year] > 0).sum()}개"
    )

print("\n===== 6개 연도 공통 사고원인 =====")
print(common_causes.to_string())

print("\n공통 사고원인 종류:", len(common_causes))

print("\n===== 공통 사고원인 건수 합계 =====")
print(common_causes.sum().to_string())

print("\n===== 공통 사고원인 포함 비율 =====")

for year in range(2020, 2026):
    total = comparison[year].sum()
    common_total = common_causes[year].sum()

    ratio = common_total / total * 100

    print(
        f"{year}년: {common_total:,} / "
        f"{total:,}건 ({ratio:.2f}%)"
    )

comparison.to_csv(
    OUTPUT_DIR / "all_year_cause_comparison.csv",
    encoding="utf-8-sig"
)

common_causes.to_csv(
    OUTPUT_DIR / "common_causes_2020_2025.csv",
    encoding="utf-8-sig"
)

print("\n분석 완료!")
