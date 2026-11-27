# Safe119 최종 UI 통합 패치

대상: 통합 브랜치 integration/safe119-v3, v3 커밋 1333655 및 XAI/자원 배분 extension 적용 상태.

기존 학습·평가 코드 및 CSV는 변경하지 않습니다. app의 여섯 화면 및 공통 UI 모듈, Streamlit 테마만 업데이트합니다. 기존 app 폴더는 적용 전에 백업합니다. 파일명은 유지하므로 기존 페이지와 중복되지 않습니다.

## 실행
프로젝트 Safe119_AI 루트에 압축을 풀고 `python -m streamlit run app/app.py --server.port 8502` 실행. 필요한 경우 기존 서버를 Ctrl+C로 종료합니다. 추가 모델 학습은 필요하지 않습니다. 이미 설치된 v3 extension 결과 CSV가 필요합니다.

## 검증
- Python 문법 검사
- 실제 v3/extension CSV를 읽고 Streamlit 입력·출력 인터페이스를 모의한 실행 검사: 6개 화면, 22개 선택 조건, Plotly 차트 75개 직렬화 성공.
- model_scores 40행, action_scores 36행: 저장 예측값에서 MAE 재계산 일치.
- 실제 Streamlit 패키지가 검증 환경에 없어 브라우저 렌더링과 위젯 동작은 사용자 PC에서 점검해야 합니다.

## 해석 원칙
후향 평가, 예측 시점의 자료 접근 가정, 실제 출동과 공개 기록의 차이, SHAP 비인과성, 가상 배분과 실제 운영의 차이를 각 화면에서 명시합니다. 기존 고정 임계값/잔차 RF 분석은 데이터 진단 탭의 탐색 실험에 남겼습니다. v3의 결과를 우선합니다. 기본 Streamlit 사이드바 탐색 구조는 호환성을 위해 유지합니다.
