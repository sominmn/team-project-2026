
# Safe119 AI

공공데이터 기반 119 구조활동 패턴 분석 및
인공지능 구조수요 예측 시스템

## 프로젝트 목표
- 2020~2025년 구조활동 데이터 분석
- 지역별 및 월별 구조활동 패턴 분석
- 머신러닝 기반 구조수요 예측
- Streamlit 대시보드 구현

## 사용 기술
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit

## 폴더 구성
- data: 원본 및 전처리 데이터
- models: AI 모델
- notebooks: 데이터 분석
- reports: 연구 보고서

v3 후향 평가 재실행

PowerShell 또는 터미널에서 Safe119_AI 디렉터리로 이동한 다음 실행합니다.

```text
python -m pip install -r requirements.txt
python diagnose_v3.py
python evaluate_v3.py
python -m streamlit run app/app.py
```

평가 규칙은 [EVALUATION_V3.md](EVALUATION_V3.md)에 있습니다. 결과는 reports/v3에 생성됩니다.
기존 진단·모델 결과와 v3 실험은 서로 다른 설정으로 계산했습니다.
새 실험은 Git에 포함된 집계 CSV를 사용합니다. 원자료 진단을 다시 계산하려면
소방청 연도별 원본을 data/raw에 넣고 audit_data_consistency.py부터 실행합니다.
