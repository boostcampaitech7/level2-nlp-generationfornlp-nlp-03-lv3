# EDA

## 데이터 구성
- 수능형 문제
    - 수능 국어, 사회 영역(윤리, 정치, 사회)
- KMMLU
    - Korean History
- MMMLU
    - HighSchool 데이터 중 역사, 경제, 정치, 지리, 심리
- KLUE-MRC
    - 경제, 교육산업, 국제, 부동산, 사회, 생활, 책마을
### **학습 데이터**
- KMMLU / MMMLU(Ko) / KLUE-MRC 데이터 중 2031개
### **평가 데이터**
- 수능형 문제 + KMMLU / MMMLU(Ko) / KLUE-MRC 데이터 총 869개
- 정답은 빈 문자열로 구성.
- 전체 **`test.csv`** 파일을 **`public.csv`**: 434개, **`private.csv`**: 435개로 랜덤하게 나누어 구성.

## Source 별 분포

<img src="https://github.com/user-attachments/assets/fd6fce6c-113e-4ca2-9427-5949dd461747" width="600" />

- KLUE-MRC: 1239개(61.03%)로 가장 높은 비중
- MMMLU: 719개(35.42%)로 두 번째로 많은 비중
- KMMLU: 72개(3.55%)로 가장 적은 비중
- 수능형 문제: 0개(0%)로 학습 데이터에는 존재하지 않음
<br>
      
## Subject 별 분포

<img src="https://github.com/user-attachments/assets/21e511ed-7033-4dc9-8e9a-f8e2f756f2db" width="600" />

- KLUE-MRC 데이터가 전체의 61.03% 차지
- world history(9.46%), us history(6.85%), european history(5.81%)로 세게사 과목이 비교적 높은 비중
- 반면 geography(0.34%), government(0.44%)과 같은 일부 과목의 비중은 매우 낮음
<br>
        
## 문장 길이 분포

| KMMLU | MMMLU | KLUE-MRC |
| --- | --- | --- |
| <img src="https://github.com/user-attachments/assets/a56c9e92-c8df-4b1b-a983-22c84dd620c8" width="600" /> | <img src="https://github.com/user-attachments/assets/0de39eb6-9ce3-4071-812f-245e9b49b905" width="600" /> | <img src="https://github.com/user-attachments/assets/a797e100-b84f-443c-991b-9c7b0244e867" width="600" /> |

- KLUE-MRC: 평균 1000 자
- MMMLU: 평균 337 자
- KMMLU: 평균 150 자
- 데이터 Source별로 문장 길이의 분포 차이가 뚜렷함
<br>

## 데이터 분석 인사이트
- 수능형 문제 미포함
  - 학습 데이터에는 수능형 문제가 포함되어 있지 않음. 따라서 수능형 문제가 테스트 데이터로 주어졌을 경우, 모델이 문제 해결에 어려움을 겪을 가능성이 큼.
  - 학습 데이터(KMMLU, MMMLU, KLUE-MRC)만으로는 특히 수능 국어 과목 문제풀이 방식을 충분히 학습하지 못할 가능성 있음.
  - 국어 과목을 비롯한 수능형 문제를 증강하여 모델이 이 유형을 학습하도록 설계 필요.
- KLUE-MRC 데이터 비중 높음
  - 학습 데이터에서 KLUE-MRC 비율이 가장 높고, Wikipedia 기반 데이터가 주를 이룸.
  - Wikipedia 기반 데이터로 학습데이터를 증강하여 모델을 훈련시킨다면 성능 개선 가능.
- 과목별 데이터 불균형
  - 학습 데이터의 subject별 분포를 보면 과목별로 서로 같은 비율을 갖지 않음. 따라서 주어진 학습 데이터만으로는 몇몇 과목의 문제 풀이 방식을 충분히 학습하지 못할 가능성이 높음.
  - 학습 데이터의 과목별 비율이 상이하므로, 증강 시 기존 분포를 고려해야 함.
