# 개요
- 학습 데이터에는 수능형 문제가 포함되어 있지 않음. 따라서 수능형 문제가 테스트 데이터로 주어졌을 경우, 모델이 문제 해결에 어려움을 겪을 가능성이 큼. 따라서 수능형 문제를 증강하여 모델이 이 유형의 문제를 잘 학습하도록 설계하는 것이 중요하다고 판단.
- 학습 데이터의 subject별 분포를 보면 과목별로 서로 같은 비율을 갖지 않음. 따라서 주어진 학습 데이터만으로는 몇몇 과목의 문제 풀이 방식을 충분히 학습하지 못할 가능성이 높음.
- 이러한 한계를 극복하기 위해 아래와 같은 데이터로 데이터셋을 증강하여 2차 Fine-Tuning을 수행.

# 검정고시
### 데이터 설명
- 원본 train 데이터셋 내 수능형 문제가 부족한 문제를 해결하기 위해 수능과 비슷한 형식의 검정고시 문항을 데이터 증강에 활용.
- 2020년부터 2024년까지의 국어, 사회, 한국사, 도덕 과목 문제들에서 총 666개를 선별하여 증강 데이터로 활용.

### 수집 방안
- PyPDF2 라이브러리와 정규표현식을 사용해 텍스트를 추출한 후 제대로 변환되지 않은 부분은 확인 후 직접 수정.

### 실험 결과
|  | Public Accuracy |  Private Accuracy |
| --- | --- | --- |
| origin | 0.6751 | 0.6207 |
| 검정고시 증강 | 0.6797 | 0.6322 |

### 결과 분석 & 인사이트
- 검정고시 문항이 성능 향상에 긍정적인 영향을 미친 것으로 판단.
<br>

# KBS한국어능력시험
### 데이터 설명
- KBS가 국민의 국어 사용 능력을 높이고 국어문화를 발전시키는 데 기여하기 위해 시행하는 시험으로 수능 국어 과목의 형식과 비슷하다고 판단되는 영역만 증강 데이터로 활용.
- 그 중 읽기 영역은 하나의 지문이 주어지고, 이에 대한 여러 개의 질문으로 구성되는 형식이 수능 국어 형식과 비슷하여 **증강 데이터로 사용**.

### 수집 방안
- 기출문제가 공개된 두 회차를 사용했고 PDF Miner를 활용하여 PDF 문서 텍스트화 작업 후 전처리 진행.
    - 이중 이상 띄어쓰기를 단일 띄어쓰기로 변경하고 불필요한 개행, 페이지 번호, 시험 정보 제거.
    - 학습 데이터 포맷화(id, paragraph, question, choices, answer, question_plus) 진행.
<img width="1380" alt="image" src="https://github.com/user-attachments/assets/e15f5cd2-fa77-423e-865a-56294141c2eb">

### 실험 결과
|  | Public Accuracy |  Private Accuracy |
| --- | --- | --- |
| origin | 0.6751 | 0.6207 |
| kbs auged | **0.6843** | **0.6253** |

### 결과 분석 & 인사이트
- 국어 문항 약 40개정도만 추가했는데도 성능이 오르는 것을 보아, 데이터 증강이 의미가 있음을 확인.
- 수능형 문제와 비슷한 문제만을 골라 증강한 것이 도움이 되었을 것으로 판단.
<br>

# [SAT Questions and Answers for LLM](https://www.kaggle.com/datasets/trainingdatapro/sat-history-questions-and-answers)
### 데이터 설명
  - SAT는 미국의 수능으로, 이 데이터는 LLM(대규모 언어 모델) 학습 및 평가를 목적으로 만들어짐.
  - 세계사(World History) 및 미국사(U.S. History) 과목 문제를 사용하였는데 이는 세계사와 미국사가 MMMLU 데이터의 비중이 높은 상위 두 과목으로, 데이터 증강 시 성능 개선에 도움이 될 것이라 판단함.

### 수집 방안
  - DeepL API활용하여 한국어로 번역
  - LLM([LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)) 을 활용한 지문 생성
    - 한국어 이해 및 생성 능력이 뛰어나며 자연스러운 문맥 처리와 정확한 텍스트 생성에서 우수한 성능을 보임.
    - 생성된 데이터 예시
      ![image](https://github.com/user-attachments/assets/dd46e05b-8a0d-4de3-b3be-2d27aca81d38)

### 실험 결과
  |  | Public Accuracy |  Private Accuracy |
  | --- | --- | --- |
  | origin | 0.6751 | 0.6207 |
  | KBS(korean) auged | 0.6843 | 0.6253 |
  | SAT(World History) auged with no paragraph | 0.6866 | **0.6322** |
  | SAT(World History) auged with paragraph | **0.6889 (최대 acc)**<br>0.6659 (최소 loss) | 0.6230 (최대 acc)<br>0.6276 (최소 loss) |
  | SAT(World History, U.S. History) auged with paragraph | 0.6636 | 0.6230 |
  
### 결과 분석 및 인사이트
  - World History (272문항)
    - 272문항을 합쳤을 때 KLUE-MRC 데이터의 개수를 넘지 않으면서, LLM이 생성한 세계사 지문을 학습함으로써 성능 <ins>향상</ins>이 있었던 것으로 판단.
  - U.S. History (1,107문항)
    - World History 과목보다 적은 비중을 차지함에도 불구하고 5배 가량의 데이터를 증강했으며, 증강 후 U.S. History 과목의 총 개수가 KLUE-MRC 데이터의 개수를 넘겨버림으로써 성능 <ins>감소</ins>가 있었던 것으로 판단.
    
# [국가공무원 시험 5급 (행정) 언어논리 데이터(PSAT)](https://www.gosi.kr/cop/bbs/selectGosiQnaList.do;jsessionid=O2Qqrjgmhd6g93YAF3sOEfcL.node_cybergosiwas21)
### 데이터 설명
  - PSAT에 수능 국어 문제와 동일하게 지문을 읽고 이해하여 올바른 선지를 고르는 유형의 문제들이 있어 증강 데이터로 사용하기로 결정.
### 수집 방안
  - 공개된 3개년 기출문제(22, 23, 24) 데이터에 대해 증강.
    <table style="width:100%;">
      <tr>
        <td style="width:50%; text-align:center;">
          <img src="https://github.com/user-attachments/assets/8d79d245-a8a4-4222-bda6-b63edfb8851a" style="max-width:100%; height:auto;">
        </td>
        <td style="width:50%; text-align:center;">
          <img src="https://github.com/user-attachments/assets/6e54e2bd-8140-4d0a-90f9-8c4172656548" style="max-width:100%; height:auto;">
        </td>
      </tr>
    </table>

    - PDF Miner를 활용해서 위와 같은 시험지를 데이터셋화.
    - answers : PDFplumber 활용해서 정답지를 표로 인식해 DataFrame으로 저장한 후 동일한 문번 index에 할당.
  - 생성된 데이터 예시
    ![image](https://github.com/user-attachments/assets/98d4448f-7a12-468e-8b3f-2083a3189ec3)
 
### 실험 결과
|  | Public Accuracy | Private Accuracy |
| --- | --- | --- |
| origin | 0.6751 | 0.6299 |
| PSAT(22, 23,24) auged | 0.6705 | 0.6230 |

### 결과 분석 및 인사이트
  - PSAT (행정) 언어논리 데이터 (94문항)
    - 2,030개의 원본데이터에서 94문항만 추가하였기 때문에, 기존 문항에 비해 증강한 데이터셋의 양이 너무 적어 큰 영향을 주지 못한것으로 보임.
 

# 공무원시험기출문제
### 데이터 설명
- 모델의 사전 지식을 보완하기 위해 공무원기출문제은행의 데이터를 활용.
- “국어”, “사회”, “한국사”, “경제학”, “교육학”, “사회복지학”에 해당하는 문제를 사용

### 수집 방안
- BeautiflSoup기반으로 휴리스틱 Parsing 알고리즘을 개발.
- 각 과목 별로 총 4가지 영역의 존재하는 모든 Text와 HTML 태그(밑줄 등)를 수집하였고, 수집된 데이터 내 존재하는 불필요한 HTML 태그를 정규표현식을 통해 제거 후 대회 학습 포맷에 맞춰 전처리하여 약 4,000개의 데이터를 수집.
- 문제의 해설은 explain 컬럼에 추가.
![image](https://github.com/user-attachments/assets/5fe5a3ac-1021-4863-8e7a-61c34a39cb23)

### 실험 결과
|  | Public Accuracy | Private Accuracy |
| --- | --- | --- |
| origin | 0.7281 | 0.7126 |
| 지문에 포함 | 0.7250 | 0.7095 |
| 지문에 포함 X | 0.7396 | 0.7149 |

### 결과 분석 & 인사이트
- explain을 지문에 포함하여 학습한 경우, 학습 성능이 <ins>떨어지는 것을</ins> 확인. 이는 지문에서 정답을 찾는 것을 학습하여 오히려 reasoing에 방해되는 것으로 판단.
- explain을 개별 column으로 따로 두어 학습한 경우, 전반적으로 모든 source에 대한 정답률이 향상되는 것을 확인.
<br>

# LLM을 활용한 데이터 증강 시도
## KorQuAD
### 데이터 설명
- KorQuAD는 한국어 기계 독해(MRC) 성능을 평가하기 위한 대표적인 한국어 데이터셋으로 주로 Wikipedia 문서에서 문단을 선택하고, 그에 맞는 질문과 답변을 작성한 데이터셋임.
- KorQuAD에서 [역사, 경제, 정치, 지리, 심리, 경제, 교육산업, 국제, 부동산, 사회, 생활, 책마을] 관련된 데이터로 데이터 증강을 시도.

### 수집 방안
- 데이터 필터링을 위해 context 중복 제거 후 unique context에 대해 **LLM**에 상기 주제와의 관련성 판단을 맡겨 유효 context에 해당하는 id를 가진 row 추출.

### 필터링 결과
- df['train'] : 60,407 → 9,041
- df['validation'] : 5,774 → 898
- 총 9,939개의 데이터를 필터링하여 학습 데이터 증강.

## 공무원시험기출문제 해설 생성
### 데이터 설명
- 학습 데이터 전체에 대해 해설(explain)을 추가 생성하여, 공무원 시험 기출 문제에서와 같이 explain 칼럼을 도입해 프롬프트에 힌트를 제공함으로써 모델 성능 향상을 기대.

### 수집 방안
- unsloth/Qwen2.5-32B-Instruct-bnb-4bit 모델을 활용한 해설 생성.
