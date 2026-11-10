
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

COMMON_CAUSES = [
    "교통", "산악", "수난", "승강기",
    "인명 갇힘", "자살추정",
    "추락", "폭발", "화재"
]

# 기존 구조활동 데이터의 지역 기준 사용
reference = pd.read_csv(
    OUTPUT_DIR / "monthly_rescue_counts.csv",
    encoding="utf-8-sig"
)

regions = sorted(reference["지역"].unique())

results = []

for year in range(2020, 2026):
    file_path = (
        RAW_DIR / f"소방청_구조활동현황_{year}1231.csv"
    )

    print(f"{year}년 처리 중...")

    for chunk in pd.read_csv(
        file_path,
        encoding="cp949",
        usecols=[
            "신고년월일",
            "발생장소_시",
            "사고원인"
        ],
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

        region = (
            chunk["발생장소_시"]
            .astype("string")
            .str.strip()
            .replace({
                "강원도": "강원특별자치도",
                "전라북도": "전북특별자치도"
            })
        )

        valid = (
            dates.notna()
            & dates.dt.year.eq(year)
            & dates.dt.month.between(1, 12)
            & causes.isin(COMMON_CAUSES)
            & region.isin(regions)
        )

        selected = pd.DataFrame({
            "연도": dates[valid].dt.year,
            "월": dates[valid].dt.month,
            "지역": region[valid]
        })

        counts = (
            selected.groupby(
                ["연도", "월", "지역"]
            )
            .size()
            .reset_index(name="구조활동건수")
        )

        results.append(counts)

df = pd.concat(results, ignore_index=True)

df = (
    df.groupby(["연도", "월", "지역"])["구조활동건수"]
    .sum()
    .reset_index()
)

# 활동이 0건인 월·지역 조합도 포함
full_index = pd.MultiIndex.from_product(
    [
        range(2020, 2026),
        range(1, 13),
        regions
    ],
    names=["연도", "월", "지역"]
)

df = (
    df.set_index(["연도", "월", "지역"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

output_path = (
    OUTPUT_DIR / "monthly_common_cause_counts.csv"
)

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 공통 사고원인 전처리 완료 =====")
print("전체 행 수:", len(df))

print("\n===== 연도별 구조활동 건수 =====")
print(
    df.groupby("연도")["구조활동건수"]
    .sum()
    .to_string()
)

print("\n저장 위치:", output_path)
