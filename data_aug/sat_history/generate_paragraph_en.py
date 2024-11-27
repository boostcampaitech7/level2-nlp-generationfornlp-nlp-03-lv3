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

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    # target_modules=["q_proj", "k_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=104,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

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

                # Enhanced prompt with more specific instructions
                prompt = f"""You are a skilled academic content writer creating a comprehensive reading passage for a standardized test.

Context: 
- Generate a factual, engaging paragraph that provides sufficient context for the following question
- The paragraph should indirectly support finding the correct answer
- Avoid directly stating the answer or making it too obvious
- Maintain academic tone and historical/factual accuracy
- Length: 150-250 words

Question: {question}
Answer Choices: {choices}
Correct Answer: {answer}

Please generate a nuanced paragraph that:
1. Provides relevant background information
2. Explores the topic without revealing the specific answer
3. Contains enough contextual details to help a test-taker reason through the options
"""

                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert academic content writer specializing in creating reading comprehension passages for standardized tests. Your goal is to craft paragraphs that provide rich context while maintaining academic integrity.",
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
                        max_new_tokens=500,
                        temperature=0.7,  # Added for more controlled generation
                        top_p=0.9,        # Added for more controlled generation
                    )
                    
                    response = tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # Extract the generated paragraph
                    generated_paragraph = response.split("<|im_start|>assistant")[-1].strip()
                    batch_outputs.append(generated_paragraph)
                except Exception as e:
                    batch_outputs.append("Paragraph generation failed")
                    print(f"Error with prompt: {prompt[:50]}... -> {e}")

            outputs.extend(batch_outputs)
            torch.cuda.empty_cache()

    df["paragraph"] = outputs
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    input_file = "sat_history_head5.csv"
    output_file = "sat_history_with_paragraph_head5.csv"
    generate_paragraph(input_file=input_file, output_file=output_file, batch_size=1, model=model, tokenizer=tokenizer)