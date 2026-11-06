
from pathlib import Path
import pandas as pd

# check_data.py가 위치한 폴더를 기준으로 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"

# 파일 이름을 변경하지 않고 자동 검색
csv_files = sorted(
    DATA_DIR.glob("소방청_구조활동현황_*.csv")
)

if not csv_files:
    print("CSV 파일을 찾을 수 없습니다.")
    print("확인할 폴더:", DATA_DIR)
    raise SystemExit

print("===== Safe119 AI 데이터 확인 =====")
print(f"발견한 CSV 파일: {len(csv_files)}개\n")

total_rows = 0

for file in csv_files:
    print("=" * 50)
    print("파일명:", file.name)

    # 메모리 절약을 위해 나눠서 읽기
    row_count = 0
    first_chunk = True

    for chunk in pd.read_csv(
        file,
        encoding="cp949",
        chunksize=50000,
        low_memory=False
    ):
        row_count += len(chunk)

        if first_chunk:
            print("컬럼 목록:")
            print(chunk.columns.tolist())

            print("\n데이터 예시:")
            print(chunk.head(3).to_string(index=False))

            first_chunk = False

    print(f"\n데이터 행 수: {row_count:,}개")
    total_rows += row_count

print("\n" + "=" * 50)
print(f"전체 데이터 행 수: {total_rows:,}개")
print("모든 파일 확인 완료!")
