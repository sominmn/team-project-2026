
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
WEATHER_DIR = BASE_DIR / "data" / "weather"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(WEATHER_DIR.glob("OBS_ASOS_TIM_*.csv"))

if len(files) != 6:
    raise FileNotFoundError(
        f"날씨 CSV 6개가 필요합니다. 현재 {len(files)}개 발견"
    )

columns = [
    "지점", "지점명", "일시",
    "기온(°C)", "강수량(mm)", "풍속(m/s)",
    "습도(%)", "적설(cm)",
    "3시간신적설(cm)", "전운량(10분위)"
]

all_monthly = []

for file in files:
    print(f"\n처리 중: {file.name}")

    # 시간별 자료를 나누어 읽기
    for chunk in pd.read_csv(
        file,
        encoding="cp949",
        usecols=columns,
        chunksize=50000
    ):
        chunk["일시"] = pd.to_datetime(
            chunk["일시"], errors="coerce"
        )

        chunk = chunk.dropna(
            subset=["지점", "일시"]
        ).copy()

        if chunk.empty:
            continue

        chunk["연도"] = chunk["일시"].dt.year
        chunk["월"] = chunk["일시"].dt.month

        # 기상 관측값을 숫자로 변환
        weather_columns = columns[3:]

        for col in weather_columns:
            chunk[col] = pd.to_numeric(
                chunk[col], errors="coerce"
            )

        # 강수량의 빈칸은 우선 결측으로 유지
        # 관측 누락과 무강수를 구분하기 위함
        chunk["강수량_합계"] = chunk["강수량(mm)"].fillna(0)

        # 월별 평균 계산을 위한 합계와 유효 관측 수
        grouped = chunk.groupby(
            ["연도", "월", "지점", "지점명"],
            as_index=False
        ).agg(
            기온합계=("기온(°C)", "sum"),
            기온관측수=("기온(°C)", "count"),
            강수량합계=("강수량_합계", "sum"),
            강수관측수=("강수량(mm)", "count"),
            풍속합계=("풍속(m/s)", "sum"),
            풍속관측수=("풍속(m/s)", "count"),
            최대풍속=("풍속(m/s)", "max"),
            습도합계=("습도(%)", "sum"),
            습도관측수=("습도(%)", "count"),
            최대적설=("적설(cm)", "max"),
            최대3시간신적설=("3시간신적설(cm)", "max"),
            구름량합계=("전운량(10분위)", "sum"),
            구름관측수=("전운량(10분위)", "count"),
            시간기록수=("일시", "count")
        )

        all_monthly.append(grouped)

print("\n월별 자료 통합 중...")

df = pd.concat(all_monthly, ignore_index=True)

# 여러 청크에 나뉜 같은 달을 다시 합치기
monthly = df.groupby(
    ["연도", "월", "지점", "지점명"],
    as_index=False
).agg(
    기온합계=("기온합계", "sum"),
    기온관측수=("기온관측수", "sum"),
    강수량합계=("강수량합계", "sum"),
    강수관측수=("강수관측수", "sum"),
    풍속합계=("풍속합계", "sum"),
    풍속관측수=("풍속관측수", "sum"),
    최대풍속=("최대풍속", "max"),
    습도합계=("습도합계", "sum"),
    습도관측수=("습도관측수", "sum"),
    최대적설=("최대적설", "max"),
    최대3시간신적설=("최대3시간신적설", "max"),
    구름량합계=("구름량합계", "sum"),
    구름관측수=("구름관측수", "sum"),
    시간기록수=("시간기록수", "sum")
)

def safe_average(total, count):
    return total.div(count.where(count > 0))

monthly["평균기온"] = safe_average(
    monthly["기온합계"], monthly["기온관측수"]
)

monthly["평균풍속"] = safe_average(
    monthly["풍속합계"], monthly["풍속관측수"]
)

monthly["평균습도"] = safe_average(
    monthly["습도합계"], monthly["습도관측수"]
)

monthly["평균구름량"] = safe_average(
    monthly["구름량합계"], monthly["구름관측수"]
)

# 강수량은 관측값 합계이며, 완전한 월 누적량인지는
# 관측 누락 여부를 확인한 후 판단해야 함
monthly = monthly.rename(columns={
    "강수량합계": "관측강수량합계"
})

result_columns = [
    "연도", "월", "지점", "지점명",
    "평균기온", "관측강수량합계",
    "평균풍속", "최대풍속",
    "평균습도", "최대적설",
    "최대3시간신적설", "평균구름량",
    "시간기록수", "기온관측수", "강수관측수"
]

result = monthly[result_columns].sort_values(
    ["연도", "월", "지점"]
)

output_path = OUTPUT_DIR / "monthly_weather_by_station.csv"

result.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 날씨 전처리 완료 =====")
print("월별 관측지점 데이터:", len(result), "행")
print("관측지점 수:", result["지점"].nunique())
print("\n처음 10행:")
print(result.head(10).to_string(index=False))
print("\n저장 위치:", output_path)
