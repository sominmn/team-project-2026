
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

for year in [2024, 2025]:
    file = RAW_DIR / f"소방청_구조활동현황_{year}1231.csv"

    print(f"\n===== {year}년 사고원인 전체 목록 =====")

    counts = pd.Series(dtype="int64")

    for chunk in pd.read_csv(
        file,
        encoding="cp949",
        usecols=["사고원인"],
        chunksize=50000
    ):
        values = (
            chunk["사고원인"]
            .astype("string")
            .str.strip()
            .fillna("미상")
        )

        counts = counts.add(
            values.value_counts(),
            fill_value=0
        )

    counts = counts.sort_values(ascending=False)

    print(counts.to_string())
    print("\n사고원인 종류 수:", len(counts))
    print("전체 건수:", int(counts.sum()))
