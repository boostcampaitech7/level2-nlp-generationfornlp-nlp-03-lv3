# 개요
- LLM([`unsloth/Qwen2.5-32B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-bnb-4bit))에 사회분야 배경지식을 주입하고, 긴 국어 지문을 읽고 정답을 추론하는 능력을 키우기 위해, 아래의 데이터셋들을 사용해서 본 train data를 학습하기 전에 1차적으로 fine tuning을 진행함.
- 영어 기반으로 사전학습된 모델이다. 따라서, TOEFL과 SAT 데이터를 영문으로 구축하여 영문 데이터에 대해 먼저 학습하고 그 후 train data를 학습하는 것으로 실험 진행.

# [TOEFL](https://github.com/iamyuanchung/TOEFL-QA/tree/master/data)
-  데이터 구조화
   - 원본 데이터
     ![image](https://github.com/user-attachments/assets/323430f2-c4e2-40d1-bac8-02be20802995)

   - 결과 데이터
     ![image](https://github.com/user-attachments/assets/75908639-f1ec-4702-82aa-464b8070bbfa)

    - 설명
      - TOEFL listening 에서 lecture에 해당하는 데이터셋 사용
      - lecture의 `SENTENCE` 에 해당하는 부분이 지문의 역할을 하기 때문에 paragraph로 사용
     
    - 증강한 데이터셋 개수
      -  684문항 증강

# [SAT](https://www.kaggle.com/datasets/trainingdatapro/sat-history-questions-and-answers)
-  데이터 구조화
   - 원본 데이터
     ![image](https://github.com/user-attachments/assets/470d15a6-c431-4f14-92fd-e19ba416b950)

   - 결과 데이터
     ![image](https://github.com/user-attachments/assets/304fd095-809f-419d-b591-4b946221e22b)

   - 설명
     - 세계사, 미국사 과목 배경 지식을 주입하기 위해 해당 데이터셋 사용
     - SAT 데이터셋은 지문 없이 오직 질문과 그에 대한 보기 선지 5개만이 주어짐을 확인.
     - LLM([`unsloth/Qwen2.5-32B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-bnb-4bit))을 사용해서 question,choices,answer 입력하고 paragraph 생성

   -  증강한 데이터셋 개수
     - 1,379 문항 증강(272-세계사, 1,107-미국역사)


# [Kowiki](https://huggingface.co/datasets/beomi/kowikitext-qa-ref-detail-preview)
-  데이터 구조화
   - 원본 데이터
     ![image](https://github.com/user-attachments/assets/71584fe2-9ed5-4a7b-8513-68b62084b2d4)
     
   - 결과 데이터

     ![image](https://github.com/user-attachments/assets/cffa5c76-5297-4e42-890c-43c7fdb34194)


   - 설명
     - 고등학교 사회탐구 영역의 각 학습목표를 가져와, 해당 학습목표를 이용해 Kowiki Filtering을 시도함.
     - 각 학습목표마다 BM25를 활용하여 30개의 가장 관련 있는 문서를 선정하여 저장함.
      ```
      # 사회문화
      "사회·문화 현상이 갖는 특성을 분석하고 다양한 관점을 적용하여 사회·문화 현상을 설명한다.",
      "사회·문화 현상을 탐구하기 위한 양적 연구 방법과 질적 연구 방법의 특징 및 차이점을 비교한다.",
      ...
      ```

   - 증강한 데이터셋 개수
     - 675문항 증강
