import subprocess

models = [
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"
]

for model in models:
    print(f"Starting training for model: {model}")
    subprocess.run(
        ["python", "train.py", f"--model_name_or_path={model}"], 
        check=True  # 실행 실패 시 에러를 던짐
    )
    print(f"Training completed for model: {model}")