import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import ast

model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")


def generate_choices(input_file, output_file, batch_size, model, tokenizer):
    df = pd.read_csv(input_file)
    outputs = []
    num_answers = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="보기 생성 중"):
            batch = df.iloc[i : i + batch_size]
            batch_outputs = []
            batch_num_answers = []

            for _, row in batch.iterrows():
                paragraph = row["context"]
                question = row["question"]
                answer = row["answers"]

                prompt = f"""
                지문: {paragraph}
                질문: {question}
                정답: {answer}

                위의 지문, 질문, 정답을 바탕으로 5개의 선지를 생성해주세요. 다음 규칙을 반드시 지켜주세요:
                1. 선지는 Python 리스트 형식으로 작성해야 합니다. 예: ['선지1', '선지2', '선지3', '선지4', '선지5']
                2. 각 선지는 한 문장으로 구성되어야 합니다.
                3. 정답은 반드시 5개의 선지 중 하나여야 합니다.
                4. 오답 선지들은 정답과 유사하지만 명확히 틀린 내용이어야 합니다.
                5. 선지들은 서로 비슷한 길이와 구조를 가져야 합니다.

                선지만 생성해주세요. 다른 설명은 필요 없습니다.
                """

                messages = [
                    {
                        "role": "system",
                        "content": "당신은 수능 문제 출제 전문가입니다. 주어진 지문, 질문, 정답을 바탕으로 적절한 선지 5개를 생성해야 합니다.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]

                try:
                    input_ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    ).to("cuda")
                    output = model.generate(
                        input_ids,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=500,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    response = tokenizer.decode(output[0], skip_special_tokens=True)
                    generated_choices = response.split("[|assistant|]")[-1].strip()

                    # 리스트 형태의 문자열을 실제 리스트로 변환
                    choices_list = ast.literal_eval(generated_choices)

                    # 정답의 인덱스 찾기
                    answer_index = choices_list.index(answer) + 1 if answer in choices_list else 0

                    batch_outputs.append(generated_choices)
                    batch_num_answers.append(answer_index)
                except Exception as e:
                    batch_outputs.append(
                        "['선지 생성 실패', '선지 생성 실패', '선지 생성 실패', '선지 생성 실패', '선지 생성 실패']"
                    )
                    batch_num_answers.append(0)
                    print(f"Error with prompt: {prompt[:50]}... -> {e}")

            outputs.extend(batch_outputs)
            num_answers.extend(batch_num_answers)
            torch.cuda.empty_cache()

    df["choices"] = outputs
    df["num_answer"] = num_answers
    df.to_csv(output_file, index=False)
    print(f"\n결과가 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    input_file = "../resources/aug/korquad/korquad_cls.csv"
    output_file = "korquad_remain_choices_ver1.csv"
    generate_choices(input_file=input_file, output_file=output_file, batch_size=1, model=model, tokenizer=tokenizer)
