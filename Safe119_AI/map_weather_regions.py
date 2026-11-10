
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
WEATHER_DIR = BASE_DIR / "data" / "weather"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# 1. 관측지점 정보 불러오기
meta_files = list(
    WEATHER_DIR.glob("META_관측지점정보_*.csv")
)

if not meta_files:
    raise FileNotFoundError("관측지점정보 CSV가 없습니다.")

# 가장 최근에 다운로드한 파일 선택
meta_path = sorted(meta_files)[-1]

meta = pd.read_csv(meta_path, encoding="cp949")

print("관측지점 정보:", meta_path.name)
print("원본 지점정보 행 수:", len(meta))

# 2. 지점번호별 최신 이력 선택
meta["시작일"] = pd.to_datetime(
    meta["시작일"], errors="coerce"
)

meta = (
    meta.sort_values("시작일")
    .drop_duplicates(subset=["지점"], keep="last")
    .copy()
)

# 3. 주소에서 시도명 추출
meta["주소"] = meta["지점주소"].astype("string").str.strip()

meta["지역"] = meta["주소"].str.split().str[0]

# 행정구역 이름 통일
region_mapping = {
    "강원도": "강원특별자치도",
    "(산지)강원도": "강원특별자치도",
    "(산지)강원특별자치도": "강원특별자치도",
    "전라북도": "전북특별자치도",
    "전남광주통합특별시": "전라남도"
}

meta["지역"] = meta["지역"].replace(region_mapping)

# 광주 관측지점(156)은 광주광역시로 구분
meta.loc[meta["지점"] == 156, "지역"] = "광주광역시"

# 4. 기존 날씨 데이터 불러오기
weather = pd.read_csv(
    PROCESSED_DIR / "monthly_weather_by_station.csv",
    encoding="utf-8-sig"
)

# 5. 지점번호로 연결
station_info = meta[["지점", "지역"]]

merged = weather.merge(
    station_info,
    on="지점",
    how="left",
    validate="many_to_one"
)

# 연결되지 않은 관측지점 확인
unmatched = merged.loc[
    merged["지역"].isna(),
    ["지점", "지점명"]
].drop_duplicates()

print("\n===== 지역 연결 확인 =====")
print("날씨 관측지점 수:", weather["지점"].nunique())
print("연결되지 않은 지점 수:", len(unmatched))

if not unmatched.empty:
    print(unmatched.to_string(index=False))
    raise ValueError(
        "지역이 연결되지 않은 지점이 있습니다."
    )

# 6. 구조활동 데이터의 17개 지역과 비교
rescue = pd.read_csv(
    PROCESSED_DIR / "monthly_rescue_counts.csv",
    encoding="utf-8-sig"
)

valid_regions = set(rescue["지역"].unique())
weather_regions = set(merged["지역"].unique())

unknown = weather_regions - valid_regions
missing = valid_regions - weather_regions

print("구조활동 데이터 지역 수:", len(valid_regions))
print("날씨 데이터 지역 수:", len(weather_regions))
print("구조활동 데이터에 없는 지역:", unknown)
print("날씨 관측지점이 없는 지역:", missing)

if unknown or missing:
    raise ValueError(
        "지역명이 일치하지 않습니다. 위 목록을 확인하세요."
    )

# 7. 월별·지역별 기상 정보 계산
# 여러 관측지점의 월별 값을 지역 평균으로 계산
# 강수량은 관측지점별 월 합계의 평균을 사용
weather_region = merged.groupby(
    ["연도", "월", "지역"],
    as_index=False
).agg(
    평균기온=("평균기온", "mean"),
    평균강수량=("관측강수량합계", "mean"),
    평균풍속=("평균풍속", "mean"),
    최대풍속=("최대풍속", "max"),
    평균습도=("평균습도", "mean"),
    최대적설=("최대적설", "max"),
    최대3시간신적설=("최대3시간신적설", "max"),
    평균구름량=("평균구름량", "mean"),
    관측지점수=("지점", "nunique")
)

weather_region = weather_region.sort_values(
    ["연도", "월", "지역"]
)

# 8. 저장
output_path = (
    PROCESSED_DIR / "monthly_weather_by_region.csv"
)

weather_region.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 지역별 날씨 데이터 생성 완료 =====")
print("전체 행 수:", len(weather_region))
print("\n처음 10행:")
print(weather_region.head(10).to_string(index=False))
print("\n저장 위치:", output_path)
