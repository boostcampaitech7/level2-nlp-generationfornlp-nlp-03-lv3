import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")


def generate_paragraph(input_file, output_file, batch_size, model, tokenizer):
    df = pd.read_csv(input_file)
    outputs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="지문 생성 중"):
            batch = df.iloc[i : i + batch_size]
            batch_outputs = []

            for _, row in batch.iterrows():
                question = row["question"]
                choices = eval(row["choices"])
                answer = row["answer"]

                prompt = f"""
                질문: {question}
                선택지: {', '.join(choices)}
                정답: {answer}

                위의 질문, 선택지, 정답을 바탕으로 적절한 지문을 생성해주세요. 지문은 질문에 답할 수 있는 충분한 정보를 포함해야 하며, 정답을 직접적으로 언급하지 않도록 주의해주세요.
                """

                messages = [
                    {
                        "role": "system",
                        "content": "당신은 역사 교과서 작가입니다. 주어진 질문, 선택지, 정답을 바탕으로 적절한 지문을 생성해야 합니다.",
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
                    )
                    response = tokenizer.decode(output[0], skip_special_tokens=True)
                    generated_paragraph = response.split("[|assistant|]")[-1].strip()
                    batch_outputs.append(generated_paragraph)
                except Exception as e:
                    batch_outputs.append("지문 생성 실패")
                    print(f"Error with prompt: {prompt[:50]}... -> {e}")

            outputs.extend(batch_outputs)
            torch.cuda.empty_cache()

    df["paragraph"] = outputs
    df.to_csv(output_file, index=False)
    print(f"\n결과가 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    input_file = "../resources/sat/sat_world_history_nonparagraph.csv"
    output_file = "../resources/sat/sat_world_history_paragraph.csv"
    generate_paragraph(input_file=input_file, output_file=output_file, batch_size=1, model=model, tokenizer=tokenizer)
