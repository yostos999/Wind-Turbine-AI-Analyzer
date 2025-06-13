# AI 설비 성능 분석가 (Wind Turbine AI Analyzer)

풍력 터빈 운영 데이터를 분석하고 AI 기반 인사이트를 제공하는 웹 애플리케이션입니다. 설비팀이 설비 성능을 쉽게 분석하고 개선 방안을 도출할 수 있도록 도와줍니다.

# 데이터 출처
https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting

## 무엇을 하는 도구인가요?

이 도구는 풍력 터빈에서 수집된 데이터(풍속, 발전량, 온도 등)를 분석해서 다음과 같은 정보를 제공합니다:

- **성능 현황**: 터빈이 얼마나 효율적으로 작동하고 있는지 분석
- **문제점 발견**: 비효율적으로 작동하는 시점과 원인 파악  
- **개선 방안**: AI가 분석한 구체적인 성능 개선 제안
- **예측 기능**: 날씨 조건에 따른 발전량 예측

## 주요 기능

### 1. 데이터 분석
- 풍속, 온도, 습도 등 환경 데이터와 발전량의 관계 분석
- 터빈의 성능 곡선 생성 및 최적 운영 구간 식별
- 비효율적으로 작동하는 시점과 패턴 자동 탐지

### 2. AI 인사이트 생성
- Google Gemini AI를 활용한 분석 결과 해석
- 상세한 성능 진단
- 즉시 실행 가능한 개선 방안 제시
- 경영진 보고용 종합 리포트 자동 생성

### 3. 예측 모델
- 랜덤포레스트 머신러닝 모델로 발전량 예측(모델 교체 가능능)
- 환경 조건(풍속, 온도, 습도)을 입력하면 예상 발전량 계산
- 실제 발전량과 예측값 비교를 통한 이상 상황 탐지

### 4. 웹 인터페이스
- 파일 업로드만으로 바로 분석 시작
- 그래프와 차트로 결과를 시각적으로 확인
- 분석 리포트를 파일로 다운로드 가능

## 누가 사용하면 좋을까요?

- **설비팀**: 설비 성능 분석 및 개선, 예방 정비 계획 수립
- **운영팀**: 일일 운영 계획 및 성과 분석
- **관리자**: 설비 투자 및 개선 의사결정

## 설치 및 실행 방법

### 1. 프로젝트 다운로드
```bash
git clone https://github.com/your-username/turbine-ai-analyzer.git
cd turbine-ai-analyzer
```

### 2. 가상환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. API 키 설정
프로젝트 폴더에 `.env` 파일을 만들고 다음 내용을 입력하세요:
```
GEMINI_API_KEY="your_gemini_api_key_here"
```

Google AI Studio(https://aistudio.google.com/)에서 API 키를 발급받을 수 있습니다.

### 5. 애플리케이션 실행
```bash
streamlit run app.py
```

웹 브라우저에서 `http://localhost:8501`로 접속하면 바로 사용할 수 있습니다.

## 사용 방법

### 1. 데이터 준비
다음 컬럼이 포함된 CSV 파일을 준비하세요:
- `Time`: 측정 시간
- `Power`: 발전량 (kW)
- `windspeed_100m`: 100m 높이 풍속 (m/s)
- `winddirection_100m`: 100m 높이 풍향 (도)
- `temperature_2m`: 2m 높이 온도 (°C)
- `relativehumidity_2m`: 2m 높이 상대습도 (%)
- `dewpoint_2m`: 2m 높이 이슬점 (°C)

### 2. 분석 실행
1. 웹 애플리케이션의 사이드바에서 데이터 파일 업로드
2. "분석 시작" 버튼 클릭
3. 분석 완료까지 대기 (보통 1-2분 소요)

### 3. 결과 확인
- **데이터 개요**: 기본 통계 및 데이터 품질 확인
- **성능 분석**: 성능 곡선과 환경 요인 간의 관계
- **효율성 진단**: 비효율 발생 현황 및 패턴
- **AI 인사이트**: AI가 생성한 분석 결과 해석
- **종합 리포트**: 최종 분석 결과 및 개선 방안

## 기술 구성

- **Python**: 데이터 분석 및 머신러닝
- **Streamlit**: 웹 인터페이스
- **Pandas & NumPy**: 데이터 처리
- **Scikit-learn**: 머신러닝 모델
- **Matplotlib & Plotly**: 데이터 시각화
- **Google Gemini**: AI 인사이트 생성

## 프로젝트 구조

```
turbine_ai_analyzer/
├── app.py                # Streamlit 웹 애플리케이션
├── config.py             # 설정 관리
├── data_analyzer.py      # 데이터 분석 엔진
├── llm_interface.py      # AI 인사이트 생성
├── utils.py              # 공통 유틸리티
├── requirements.txt      # 필요 패키지 목록
├── .env                  # API 키 설정
└── data/
    └── Location1.csv    # 샘플 데이터
```

## 인터페이스
![스크린샷 2025-06-13 220430](https://github.com/user-attachments/assets/5d3c7be3-9225-470a-a476-ec6c9cd69fb1)
기본 화면입니다.

![스크린샷 2025-06-13 220612](https://github.com/user-attachments/assets/af8d697a-9b31-473b-9306-c4b6ca8d1f86)
csv파일을 업로드 하시거나 또는 셈플 데이터를 사용한 다음 분석 시작을 누르면 위 화면과 같이 나옵니다.

![스크린샷 2025-06-13 220636](https://github.com/user-attachments/assets/42033690-8d52-4543-8f12-83bf8e1a039d)
성능 분석 창입니다.

![스크린샷 2025-06-13 220644](https://github.com/user-attachments/assets/457134b8-d02a-46c2-ba83-f0f33f6da656)
![스크린샷 2025-06-13 220650](https://github.com/user-attachments/assets/71fe70cd-9b30-4069-b9e3-0c39e5dc115f)
상관관계에 대한 히트맵 및 발전량과의 상관관계를 보여주는 그래프입니다.

![스크린샷 2025-06-13 220714](https://github.com/user-attachments/assets/b5f9b277-c3cb-4d5b-8962-0122edc79d1a)
효율성 진단 창입니다.

![스크린샷 2025-06-13 220826](https://github.com/user-attachments/assets/dcefa732-03a1-4070-97e7-a81081e93fec)
AI 인사이트 생성을 누르면 제미나이 API(적용 모델은 1.5 flash)를 활용하여 데이터를 분석하고 인사이트를 만들어 줍니다.

![스크린샷 2025-06-13 221024](https://github.com/user-attachments/assets/72e4f969-03c8-4b15-b628-b819d73eaadc)
![스크린샷 2025-06-13 220854](https://github.com/user-attachments/assets/90b8a28b-1c80-4aac-8644-4cd50dcc9a52)
![스크린샷 2025-06-13 220839](https://github.com/user-attachments/assets/82e2fa33-3482-4a97-af78-0362e1dffaa9)
또한 추가적인 조언을 포함한 리포트를 생성해 줍니다.

## 라이선스

MIT License - 자유롭게 사용, 수정, 배포가 가능합니다.

## 문의사항

프로젝트 사용 중 문제가 발생하거나 개선 아이디어가 있으시면 GitHub Issues를 통해 연락주세요.
