
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

common_causes = [
    "교통", "산악", "수난", "승강기",
    "인명 갇힘", "자살추정", "추락",
    "폭발", "화재"
]

results = []

for year in [2024, 2025]:
    file_path = (
        RAW_DIR / f"소방청_구조활동현황_{year}1231.csv"
    )

    print(f"{year}년 분석 중...")

    for chunk in pd.read_csv(
        file_path,
        encoding="cp949",
        usecols=["신고년월일", "사고원인"],
        chunksize=50000
    ):
        dates = pd.to_datetime(
            chunk["신고년월일"],
            errors="coerce"
        )

        causes = (
            chunk["사고원인"]
            .astype("string")
            .str.strip()
        )

        valid = (
            dates.notna()
            & dates.dt.year.eq(year)
            & causes.isin(common_causes)
        )

        selected = pd.DataFrame({
            "월": dates[valid].dt.month,
            "사고원인": causes[valid]
        })

        counts = (
            selected.groupby(["월", "사고원인"])
            .size()
            .reset_index(name="건수")
        )

        counts["연도"] = year
        results.append(counts)

df = pd.concat(results, ignore_index=True)

monthly = (
    df.groupby(["연도", "월"])["건수"]
    .sum()
    .unstack("연도", fill_value=0)
    .reindex(range(1, 13), fill_value=0)
)

monthly["증감"] = monthly[2025] - monthly[2024]

monthly["증감률(%)"] = (
    monthly["증감"]
    .div(monthly[2024].replace(0, float("nan")))
    .mul(100)
    .round(1)
)

print("\n===== 공통 사고원인 월별 비교 =====")
print(monthly.to_string())

cause_summary = (
    df.groupby(["사고원인", "연도"])["건수"]
    .sum()
    .unstack("연도", fill_value=0)
)

cause_summary["증감"] = (
    cause_summary[2025] - cause_summary[2024]
)

print("\n===== 사고원인별 증감 =====")
print(
    cause_summary.sort_values(
        "증감", ascending=False
    ).to_string()
)

monthly.to_csv(
    OUTPUT_DIR / "common_monthly_comparison.csv",
    encoding="utf-8-sig"
)

print("\n분석 완료!")
