
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "monthly_rescue_counts.csv"
)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# 1. 연도별 총 구조활동 건수
print("\n===== 연도별 구조활동 건수 =====")
print(
    df.groupby("연도")["구조활동건수"]
    .sum()
    .to_string()
)

# 2. 연도별 월간 구조활동 건수
print("\n===== 연도별·월별 구조활동 건수 =====")

pivot = df.pivot_table(
    index="월",
    columns="연도",
    values="구조활동건수",
    aggfunc="sum"
)

print(pivot.to_string())

# 3. 월별 지역 수 확인
print("\n===== 연도별·월별 지역 수 =====")

region_count = df.pivot_table(
    index="월",
    columns="연도",
    values="지역",
    aggfunc="nunique"
)

print(region_count.to_string())

# 4. 데이터 검증
print("\n===== 데이터 검증 =====")
print("전체 지역 수:", df["지역"].nunique())
print("누락된 값:", df.isna().sum().sum())
print("중복된 연도·월·지역:", df.duplicated(
    subset=["연도", "월", "지역"]
).sum())
