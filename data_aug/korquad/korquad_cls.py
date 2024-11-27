import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_model_locally(path="../../saved_model"):
    """모델과 토크나이저를 메모리 효율적으로 로드"""
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,  # 더 가벼운 데이터 타입 사용
        trust_remote_code=True,
        device_map="auto",  # 자동 장치 매핑
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def classify_domain(input_file, output_file, batch_size, model, tokenizer):
    """주어진 wiki datasets에 대해 사회 도메인 여부를 분류"""
    df = pd.read_csv(input_file)
    datasets = df["context"].tolist()
    outputs = []

    with torch.no_grad():  # 예측 과정 전체에서 메모리 사용 제한
        for i in tqdm(range(0, len(datasets), batch_size), desc="도메인 분류 진행 중"):
            batch = datasets[i : i + batch_size]
            batch_outputs = []

            for prompt in batch:
                messages = [
                    {
                        "role": "system",
                        "content": """당신은 수능 사회탐구 영역 문제 풀이 모델을 위한 데이터 필터링 어시스턴트입니다.
                         사회지식 데이터를 만들기 위해, 수능 사회탐구 영역과 관련된 주제에 맞지 않는 텍스트를 필터링해야 합니다.
                         수능 사회탐구 영역과 관련된 주제는 **역사, 경제, 정치, 지리, 심리, 경제, 교육산업, 국제, 부동산, 사회, 생활, 책마을**이 있습니다.

                         주제 평가 기준:
                         - 지문이 주어지는 평가 주제를 다루는 경우 '유효'로 분류
                         - 지문이 주어지는 평가 주제를 벗어나는 주제를 다루는 경우 '제거'로 분류

                         불확실한 경우:
                         - 지문의 사회탐구 관련성이 모호하다면 '불확실' 선택

                         규칙:
                         1. 반환값은 오직 '유효', '제거', '불확실' 중 하나
                         2. 부가 설명 없이 결과만 제시""",
                    },
                    {
                        "role": "user",
                        "content": f"주어지는 지문을 보고 주제 평가 기준에 맞추어 '유효', '제거', '불확실' 중 하나로 답해주세요.\n{prompt}",
                    },
                ]

                try:
                    input_ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    ).to("cuda")

                    output = model.generate(
                        input_ids,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=2,
                    )

                    # '[|assistant|]' 뒤에 오는 값만 추출
                    response = tokenizer.decode(output[0], skip_special_tokens=True)
                    user_response = response.split("[|assistant|]")[-1].strip()

                    if "유효" in user_response:
                        batch_outputs.append("유효")
                    elif "제거" in user_response:
                        batch_outputs.append("제거")
                    elif "불확실" in user_response:
                        batch_outputs.append("불확실")
                    else:
                        batch_outputs.append("분류불가")
                except Exception as e:
                    batch_outputs.append("분류불가")
                    print(f"Error with prompt: {prompt[:50]}... -> {e}")

            outputs.extend(batch_outputs)
            torch.cuda.empty_cache()  # 배치 처리 후 GPU 메모리 해제

    df["cls"] = outputs

    # 결과를 CSV 파일로 저장
    df.to_csv(output_file, index=False)
    print(f"\n결과가 {output_file}에 저장되었습니다.")

    # 분류 결과 통계 출력
    print("\n분류 결과 통계:")
    print(df["cls"].value_counts())

if __name__ == "__main__":
    input_file = "korquad_train.csv"
    output_file = "korquad_train_cls.csv"
    model, tokenizer = load_model_locally()
    classify_domain(input_file=input_file, output_file=output_file, batch_size=1, model=model, tokenizer=tokenizer)