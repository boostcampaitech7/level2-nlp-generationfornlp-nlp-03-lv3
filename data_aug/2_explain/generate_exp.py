import pandas as pd
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm


# 모델 및 토크나이저 불러오기
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    dtype=None,
    load_in_4bit=True,
    device_map="auto",
)
FastLanguageModel.for_inference(model)


def generate_paragraph(input_file, output_file, batch_size, model, tokenizer):
    df = pd.read_csv(input_file)
    outputs = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating Explain"):
        paragraph = row["paragraph"]
        question = row["question"]
        question_plus = row["question_plus"]
        choices = row["choices"]
        # Enhanced prompt with more specific instructions
        prompt = f"""당신은 전문적인 교육 콘텐츠 개발자로, 표준화된 시험의 문제 해설을 작성하고 있습니다.
        
        지문: {paragraph}
        보기: {question_plus}
        문제: {question}
        선택지: {choices}
        
        해설 작성 지침:
        1. 지문의 핵심 내용과 문제의 연관성을 명확히 분석하세요.
        2. 각 선택지를 지문의 정보와 연결하여 논리적으로 검토하세요.
        3. 정답을 도출하는 데에 대한 추론 과정을 설명하세요.
        4. 학술적이고 객관적인 어조를 유지하세요.
        5. 해설의 길이는 100단어 이하로 작성하세요.
        
        사전 지식 고려사항:
        - 문제 해결에 필요한 배경 지식이 있다면 반드시 상세히 설명하세요.
        - 사전 지식의 핵심 개념, 역사적 맥락, 또는 기본 원리를 명확하게 제시하세요.
        - 학생들이 문제를 이해하는 데 필요한 추가 정보를 제공하세요.

        해설에 포함해야 할 사항:
        - 지문의 주요 논점
        - 각 선택지의 타당성 분석
        - 올바른 추론 방식
        - 핵심적인 맥락적 단서들
        - 문제 해결에 필수적인 사전 지식 및 개념 설명
        
        요구사항:
        - 학생들이 사고 과정을 이해할 수 있도록 명확하고 체계적인 설명
        - 지문의 세부 사항을 활용한 논리적 추론 제시
        - 사전지식이 필요한 경우, 그 내용을 상세하고 이해하기 쉽게 추가
        - 단순 암기가 아닌 비판적 사고를 유도하는 해설
        """

        messages = [
            {
                "role": "system",
                "content": "당신은 표준화된 시험 문제의 해설을 전문적으로 작성하는 교육 전문가입니다. 학생들의 추론 능력을 촉진하는 해설을 만드는 것이 목표입니다.",
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
            # Adjusted generation parameters for more controlled output
            output = model.generate(
                input_ids,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=256,
                temperature=0.7,  # Added for more controlled generation
                top_p=0.9,  # Added for more controlled generation
            )

            response = tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract the generated paragraph
            generated_paragraph = response.split("assistant")[-1].strip()
            print(generated_paragraph)
            outputs.append(generated_paragraph)
        except Exception as e:
            outputs.append("Paragraph generation failed")
            print(f"Error with prompt: {prompt[:50]}... -> {e}")
        torch.cuda.empty_cache()

    df["explain"] = outputs
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    input_file = "../resources/merge/merge_dataset_20241125.csv"
    df = pd.read_csv(input_file)
    output_file = "../resources/merge/merge_dataset_20241125_exp.csv"
    generate_paragraph(input_file=input_file, output_file=output_file, batch_size=1, model=model, tokenizer=tokenizer)
