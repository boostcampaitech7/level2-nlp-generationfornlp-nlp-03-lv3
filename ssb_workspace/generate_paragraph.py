import pandas as pd
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm


# 모델 및 토크나이저 불러오기
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    dtype=None,
    load_in_4bit=True,
    device_map="auto",
)
FastLanguageModel.for_inference(model)

def generate_paragraph(input_file, output_file, batch_size, model, tokenizer):
    df = pd.read_csv(input_file)
    outputs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Generating Paragraphs"):
            batch = df.iloc[i : i + batch_size]
            batch_outputs = []

            for _, row in batch.iterrows():
                question = row['question']
                choices = row['choices']
                answer = row['answer']
                explain = row['explain']
                # Enhanced prompt with more specific instructions
                prompt = f"""당신은 수준 높은 학술 콘텐츠 작성자로, 표준화된 시험을 위한 독해 지문을 만들고 있습니다.

                맥락:
                - 다음 문제에 대한 사실적이고 흥미로운 단락을 생성하세요
                - 지문은 간접적으로 정답을 찾는 데 도움을 줘야 합니다
                - 답을 직접적으로 드러내거나 너무 명백하게 만들지 마세요
                - 학술적 어조와 역사적/사실적 정확성을 유지하세요

                문제: {question}
                선택지: {choices}
                정답: {answer}
                해설: {explain}

                다음 사항을 고려하여 세밀한 단락을 생성하세요:
                1. 관련된 배경 정보 제공
                2. 특정 답을 드러내지 않고 주제 탐구
                3. 수험생이 선택지를 추론할 수 있을 만큼 충분한 맥락적 세부 정보 포함
                """

                messages = [
                    {
                        "role": "system",
                        "content": "당신은 표준화된 시험의 독해 지문을 전문적으로 작성하는 학술 콘텐츠 작성자입니다. 목표는 학문적 진실성을 유지하면서 풍부한 맥락을 제공하는 단락을 만드는 것입니다.",
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
                        max_new_tokens=1024,
                        temperature=0.7,  # Added for more controlled generation
                        top_p=0.9,        # Added for more controlled generation
                    )
                    
                    response = tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # Extract the generated paragraph
                    generated_paragraph = response.split("assistant")[-1].strip()
                    batch_outputs.append(generated_paragraph)
                except Exception as e:
                    batch_outputs.append("Paragraph generation failed")
                    print(f"Error with prompt: {prompt[:50]}... -> {e}")

            outputs.extend(batch_outputs)
            torch.cuda.empty_cache()

    df["paragraph"] =outputs
    df.to_csv(output_file, index=False,encoding='utf-8-sig')
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    input_file = "crawled_nonparagraph_2.csv"
    output_file = "crawled_nonparagraph_output_2.csv"
    generate_paragraph(input_file=input_file, output_file=output_file, batch_size=1, model=model, tokenizer=tokenizer)