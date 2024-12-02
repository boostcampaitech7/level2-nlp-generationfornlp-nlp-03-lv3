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
 <br>
 
## Source 별 분포

<img src="https://github.com/user-attachments/assets/fd6fce6c-113e-4ca2-9427-5949dd461747" width="600" />
<br>
      
## Subject 별 분포

<img src="https://github.com/user-attachments/assets/21e511ed-7033-4dc9-8e9a-f8e2f756f2db" width="600" />
<br>
        
## 문장 길이 분포

| KMMLU | MMMLU | KLUE-MRC |
| --- | --- | --- |
| <img src="https://github.com/user-attachments/assets/a56c9e92-c8df-4b1b-a983-22c84dd620c8" width="600" /> | <img src="https://github.com/user-attachments/assets/0de39eb6-9ce3-4071-812f-245e9b49b905" width="600" /> | <img src="https://github.com/user-attachments/assets/a797e100-b84f-443c-991b-9c7b0244e867" width="600" /> |
<br>

## 데이터 분석 인사이트
- 수능형 문제 미포함
  - 학습 데이터(KMMLU, MMMLU, KLUE-MRC)에 수능형 문제가 없어 국어 과목 문제풀이 방식을 충분히 학습하지 못할 가능성 있음.
  - 수능형 문제를 증강해 모델이 이 유형을 학습하도록 설계 필요.
- KLUE-MRC 데이터 비중 높음
  - 학습 데이터에서 KLUE-MRC 비율이 가장 높고, Wikipedia 기반 데이터가 주를 이룸.
  - Wikipedia 기반 데이터를 사전학습에 활용 시 성능 개선 가능.
- 과목별 데이터 불균형
  - 학습 데이터의 과목별 비율이 상이하므로, 증강 시 기존 분포를 고려해야 함.
- 문장 길이 분포 차이
  - Source별 문장 길이 분포가 다르므로, 길이에 따라 RAG 활용 여부를 구분 가능.
