
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = range(2020, 2026)
CHUNK_SIZE = 50000

CAUSE = "사고원인"
DETAIL = "사고원인코드명_사고종별"
DATE = "신고년월일"
REGION = "발생장소_시"

year_stats = []
cause_counts = defaultdict(Counter)
detail_counts = defaultdict(Counter)
monthly_counts = defaultdict(Counter)
columns_by_year = {}

def clean_text(series):
    return (
        series.fillna("(결측)")
        .astype(str)
        .str.strip()
        .replace("", "(결측)")
    )

for year in YEARS:
    path = RAW / f"소방청_구조활동현황_{year}1231.csv"

    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    print(f"\n===== {year}년 원본 검사 =====")

    total = 0
    missing_cause = 0
    missing_detail = 0
    invalid_date = 0
    wrong_year = 0
    missing_region = 0

    reader = pd.read_csv(
        path,
        encoding="cp949",
        chunksize=CHUNK_SIZE,
        dtype=str,
        low_memory=False
    )

    for chunk in reader:
        if year not in columns_by_year:
            columns_by_year[year] = set(chunk.columns)

        required = [CAUSE, DETAIL, DATE, REGION]
        missing = [c for c in required if c not in chunk.columns]

        if missing:
            raise ValueError(
                f"{year}년 필수 컬럼 없음: {missing}"
            )

        total += len(chunk)

        causes = clean_text(chunk[CAUSE])
        details = clean_text(chunk[DETAIL])
        regions = clean_text(chunk[REGION])

        missing_cause += (causes == "(결측)").sum()
        missing_detail += (details == "(결측)").sum()
        missing_region += (regions == "(결측)").sum()

        dates = pd.to_datetime(
            chunk[DATE],
            errors="coerce",
            format="mixed"
        )

        invalid_date += dates.isna().sum()

        valid = dates.notna()
        wrong_year += (
            valid & (dates.dt.year != year)
        ).sum()

        # 원본 분류 분포: 날짜 오류와 무관하게 집계
        cause_counts[year].update(causes.tolist())

        pairs = zip(causes.tolist(), details.tolist())
        detail_counts[year].update(pairs)

        # 월별 집계: 해당 연도의 유효 날짜만 사용
        valid_year = valid & (dates.dt.year == year)
        month_values = dates.loc[valid_year].dt.month
        monthly_counts[year].update(
            month_values.value_counts().to_dict()
        )

    year_stats.append({
        "연도": year,
        "원본행수": total,
        "사고원인결측": int(missing_cause),
        "세부유형결측": int(missing_detail),
        "날짜파싱실패": int(invalid_date),
        "다른연도날짜": int(wrong_year),
        "지역결측": int(missing_region),
        "사고원인종류": len(cause_counts[year]),
        "세부유형조합종류": len(detail_counts[year])
    })

    print(f"원본 행 수: {total:,}")
    print(f"사고원인 종류: {len(cause_counts[year])}")
    print(f"세부유형 조합: {len(detail_counts[year])}")
    print(f"날짜 파싱 실패: {invalid_date:,}")
    print(f"다른 연도 날짜: {wrong_year:,}")

# 1. 연도별 품질
quality = pd.DataFrame(year_stats)
quality.to_csv(
    OUT / "audit_year_quality.csv",
    index=False,
    encoding="utf-8-sig"
)

# 2. 연도별 컬럼 비교
all_columns = sorted(set.union(*columns_by_year.values()))
schema = pd.DataFrame([
    {
        "컬럼명": col,
        **{
            str(year): int(col in columns_by_year[year])
            for year in YEARS
        }
    }
    for col in all_columns
])
schema.to_csv(
    OUT / "audit_schema.csv",
    index=False,
    encoding="utf-8-sig"
)

# 3. 사고원인별 연도별 건수
all_causes = sorted(
    set().union(
        *(set(cause_counts[y]) for y in YEARS)
    )
)

cause_table = pd.DataFrame([
    {
        "사고원인": cause,
        **{
            str(y): cause_counts[y][cause]
            for y in YEARS
        }
    }
    for cause in all_causes
])

cause_table.to_csv(
    OUT / "audit_cause_counts.csv",
    index=False,
    encoding="utf-8-sig"
)

# 4. 2024년에만 있거나 2025년에만 있는 분류
old = set(cause_counts[2024])
new = set(cause_counts[2025])

removed = sorted(old - new)
added = sorted(new - old)
common = sorted(old & new)

print("\n===== 2024 → 2025 사고원인 비교 =====")
print("2024년에만 존재:", removed)
print("2025년에만 존재:", added)
print("공통 사고원인:", common)

# 5. 공통 사고원인의 세부유형 변화
detail_rows = []

for cause in common:
    old_details = {
        d for c, d in detail_counts[2024]
        if c == cause
    }
    new_details = {
        d for c, d in detail_counts[2025]
        if c == cause
    }

    detail_rows.append({
        "사고원인": cause,
        "2024세부유형수": len(old_details),
        "2025세부유형수": len(new_details),
        "2024에만있는세부유형": " | ".join(
            sorted(old_details - new_details)
        ),
        "2025에만있는세부유형": " | ".join(
            sorted(new_details - old_details)
        ),
        "공통세부유형수": len(
            old_details & new_details
        )
    })

detail_table = pd.DataFrame(detail_rows)
detail_table.to_csv(
    OUT / "audit_detail_changes.csv",
    index=False,
    encoding="utf-8-sig"
)

# 6. 월별 원본 기록 수 비교
monthly_rows = []

for year in YEARS:
    for month in range(1, 13):
        monthly_rows.append({
            "연도": year,
            "월": month,
            "유효날짜기록수": monthly_counts[year][month]
        })

pd.DataFrame(monthly_rows).to_csv(
    OUT / "audit_monthly_counts.csv",
    index=False,
    encoding="utf-8-sig"
)

# 7. 연도별 분류 구성비
share_rows = []

for year in YEARS:
    total = sum(cause_counts[year].values())

    for cause, count in cause_counts[year].items():
        share_rows.append({
            "연도": year,
            "사고원인": cause,
            "건수": count,
            "구성비": count / total if total else 0
        })

pd.DataFrame(share_rows).to_csv(
    OUT / "audit_cause_shares.csv",
    index=False,
    encoding="utf-8-sig"
)

# 8. 핵심 변화 출력
print("\n===== 2024년에는 있고 2025년에는 없는 분류 =====")

for cause in removed:
    print(
        f"{cause}: "
        f"2024년 {cause_counts[2024][cause]:,}건"
    )

print("\n===== 공통 분류의 건수 변화 =====")

for cause in common:
    a = cause_counts[2024][cause]
    b = cause_counts[2025][cause]
    change = ((b - a) / a * 100) if a else float("nan")

    print(
        f"{cause}: {a:,} → {b:,} "
        f"({change:+.1f}%)"
    )

print("\n===== 세부유형이 달라진 공통 분류 =====")

changed = detail_table[
    (detail_table["2024에만있는세부유형"] != "")
    | (detail_table["2025에만있는세부유형"] != "")
]

print(
    changed[
        ["사고원인", "2024세부유형수",
         "2025세부유형수", "공통세부유형수"]
    ].to_string(index=False)
)

print("\n===== 연도별 데이터 품질 =====")
print(quality.to_string(index=False))

print("\n검증 완료! audit_*.csv 파일 저장됨.")
