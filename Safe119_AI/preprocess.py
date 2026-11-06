
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.glob("소방청_구조활동현황_*.csv"))

if not files:
    raise FileNotFoundError("원본 CSV 파일을 찾을 수 없습니다.")

all_monthly = []
total_valid = 0
total_invalid = 0

for file in files:
    print(f"\n처리 중: {file.name}")

    for chunk in pd.read_csv(
        file,
        encoding="cp949",
        usecols=["신고년월일", "발생장소_시"],
        chunksize=50000
    ):
        # 날짜 변환
        chunk["신고년월일"] = pd.to_datetime(
            chunk["신고년월일"],
            errors="coerce"
        )

        # 지역명 정리
        chunk["발생장소_시"] = (
            chunk["발생장소_시"]
            .astype("string")
            .str.strip()
        )

        # 날짜 또는 지역이 없는 행 제외
        valid = chunk.dropna(
            subset=["신고년월일", "발생장소_시"]
        ).copy()

        valid = valid[valid["발생장소_시"] != ""]

        total_valid += len(valid)
        total_invalid += len(chunk) - len(valid)

        if valid.empty:
            continue

        valid["연도"] = valid["신고년월일"].dt.year
        valid["월"] = valid["신고년월일"].dt.month
        valid["지역"] = valid["발생장소_시"]

        monthly = (
            valid.groupby(
                ["연도", "월", "지역"]
            )
            .size()
            .reset_index(name="구조활동건수")
        )

        all_monthly.append(monthly)

# 여러 파일과 청크에서 계산한 건수 합산
result = pd.concat(all_monthly, ignore_index=True)

result = (
    result.groupby(["연도", "월", "지역"], as_index=False)
    ["구조활동건수"]
    .sum()
    .sort_values(["연도", "월", "지역"])
)

# 분석용 CSV 저장
output_path = OUTPUT_DIR / "monthly_rescue_counts.csv"

result.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 전처리 완료 =====")
print(f"유효한 기록: {total_valid:,}건")
print(f"제외된 기록: {total_invalid:,}건")
print(f"월별 집계 행 수: {len(result):,}행")

print("\n처음 10행:")
print(result.head(10).to_string(index=False))

print(f"\n저장 위치: {output_path}")
