
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(
    DATA_DIR / "rescue_weather_merged.csv",
    encoding="utf-8-sig"
)

weather_columns = [
    "평균기온",
    "평균강수량",
    "평균풍속",
    "최대풍속",
    "평균습도",
    "최대적설",
    "최대3시간신적설",
    "평균구름량"
]

print("\n===== 날씨 변수별 결측률 =====")

missing = (
    df[weather_columns]
    .isna()
    .mean()
    .mul(100)
    .round(2)
    .sort_values(ascending=False)
)

print(missing.to_string())

print("\n===== 지역별 관측지점 수 =====")

station_count = (
    df.groupby("지역")["관측지점수"]
    .agg(["min", "max", "mean"])
    .round(2)
)

print(station_count.to_string())

print("\n===== 연도별 날씨 데이터 결측률 =====")

yearly_missing = (
    df.groupby("연도")[weather_columns]
    .apply(lambda x: x.isna().mean() * 100)
    .round(2)
)

print(yearly_missing.to_string())

print("\n===== 월별 강수량 범위 =====")
print(
    df.groupby("연도")["평균강수량"]
    .agg(["min", "max", "mean"])
    .round(2)
    .to_string()
)

print("\n===== 검증 완료 =====")
