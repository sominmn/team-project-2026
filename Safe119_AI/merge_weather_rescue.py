
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

rescue = pd.read_csv(
    DATA_DIR / "monthly_rescue_counts.csv",
    encoding="utf-8-sig"
)

weather = pd.read_csv(
    DATA_DIR / "monthly_weather_by_region.csv",
    encoding="utf-8-sig"
)

print("구조활동 데이터:", len(rescue), "행")
print("날씨 데이터:", len(weather), "행")

keys = ["연도", "월", "지역"]

# 중복 키 확인
if weather.duplicated(keys).any():
    raise ValueError("날씨 데이터에 중복된 월·지역이 있습니다.")

# 구조활동 데이터를 기준으로 병합
merged = rescue.merge(
    weather,
    on=keys,
    how="left",
    validate="one_to_one",
    indicator=True
)

print("\n===== 병합 결과 =====")
print(merged["_merge"].value_counts().to_string())

# 날씨 데이터가 연결되지 않은 지역 확인
unmatched = merged[merged["_merge"] != "both"]

if not unmatched.empty:
    print("\n날씨가 연결되지 않은 행:")
    print(unmatched[keys].to_string(index=False))
    raise ValueError("날씨 데이터 병합이 불완전합니다.")

merged = merged.drop(columns=["_merge"])

output_path = DATA_DIR / "rescue_weather_merged.csv"

merged.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 병합 완료 =====")
print("전체 행 수:", len(merged))
print("전체 열 수:", len(merged.columns))
print("\n처음 10행:")
print(merged.head(10).to_string(index=False))
print("\n저장 위치:", output_path)
