### K-Fold<br>
- K-Fold 개념
    - 전체 데이터셋을 k개의 fold로 분할한 후, 매 학습마다 하나의 fold는 validation, 나머지는 train 데이터로 사용하는 방식.
     ![388376949-d02aae68-9715-4f01-970d-82bc6396c862](https://github.com/user-attachments/assets/78f3ad63-bca4-45a3-b85d-eb159c5240bc)

- 사용 목적
    - train/validation 데이터 분할 시 샘플링이 최대한 골고루 되도록 함
    - 제한된 양의 데이터를 최대한 활용하여 모든 데이터를 학습에 이용할 수 있도록 함

1. Random K-Fold : 검정고시 증강 데이터를 5개의 fold로 랜덤 분할
      - fold 안에 국어, 사회 문제가 골고루 들어가도록 처리를 하지 않았음에도 public accuracy 기준 약 4%의 성능 향상이 확인됨.
        
      | | Public Accuracy | Private Accuracy |
      |---|---|---|
      | origin | 0.6774 | 0.6184 |
      | 5 fold + ensemble (hard voting) | 0.7143 | 0.6805 |
  
2. Stratified K-Fold : 데이터를 국어/사회 분야로 나눈 후 각 fold가 전체 데이터의 국어/사회 비율을 유지하도록 함.
      - fold 구성
        - 국어 분야: 검정고시 및 공무원 시험 국어 과목, KLUE-MRC 문제
        - 사회 분야: 나머지 문제들
      - Public accuracy 기준 약 9%의 성능 향상이 있었음.
      - 두 실험 간 fold 수와 데이터셋 구성에 차이가 있으나, 성능 향상 폭을 고려할 때 데이터를 균형적으로 분할하는 것이 모델 성능 개선에 더 큰 영향을 미친 것으로 판단됨.

      | | Public Accuracy | Private Accuracy |
      |---|---|---|
      | origin | 0.7143 | 0.6805 |
      | 4 fold + ensemble (hard voting) | 0.8065 | 0.7770 |
      | 4 fold + ensemble (soft voting) | 0.8088 | 0.7747 |
      
### 앙상블
- 배경
    - 데이터 증강 및 K-Fold로 성능을 끌어올리는 데 한계가 있어 더 좋은 성능을 얻기 위해 여러 개의 모델을 결합하여 더 좋은 성능을 얻는 앙상블 기법을 도입.
- 방법론
    - Soft Voting
        - 각 선택지 별 확률값의 평균을 계산하여 가장 높은 확률을 가진 선택지를 최종 답안으로 선정.
    - Soft Voting ver2
        - Soft Voting 결과 답안으로 선택되는 선택지의 확률 값이 점점 낮아지는 현상이 있음.
        - 모델의 확신도가 높을수록 정답일 확률이 높아질 것이라는 가정에 따라 Soft Voting ver2를 제작함.
        - 평균 확률값이 임계값 미만인 경우, 개별 모델 중 가장 높은 확신도를 보인 답안을 선택하는 방식으로 보완.
    - Hard Voting
        - 다수결 방식으로 가장 많은 투표를 받은 선택지를 최종 답안으로 선정.
        - 동점이 발생한 경우, 동점인 선택지들 중에서 가장 높은 확률값을 가진 것을 최종 답안으로 선택.
    - Hard Voting ver2
        - 기본 hard voting과 비교했을 때 동점 발생 시 처리 방식에서 차이가 있음.
        - 동점인 선택지들에 대해 각 모델이 예측한 확률값들의 합이 가장 큰 선택지를 최종 답안으로 선택.
- 실험 결과 및 분석
    - 첫번째 실험에서 Public Accuracy 기준 Soft Voting이 Hard Voting보다 결과가 좋음.
    - 두번째 실험에서 soft voting ver2가 성능이 좋았음.
    - 실험결과에 따라 soft voting ver2의 성능이 가장 좋을 것이라 가정하고, public accuracy가 높았던 여러 결과들에대해 Soft Voting ver2로 앙상블을 수행함.
    - 실험 결과
        - 앙상블한 데이터셋 : 검정고시 증강 데이터 fold 4개
        
        |  | Public Accuracy |  Private Accuracy |
        | --- | --- | --- |
        | origin | 0.7143 | 0.6805 |
        | 4 fold + ensemble (Hard Voting) | 0.8065 | 0.7770 |
        | 4 fold + ensemble (Soft Voting) | 0.8088 | 0.7747 |
        - 앙상블한 데이터셋 : merge_dataset_20241124_fold_1 학습 결과 + 검정고시 증강 데이터 4 fold 후 soft voting한 결과 + toefl, sat 데이터 학습 결과
        
        |  | Public Accuracy |  Private Accuracy |
        | --- | --- | --- |
        | Soft Voting | 0.8088 | 0.7724 |
        | Soft Voting ver2 | 0.8134 | 0.7747 |
