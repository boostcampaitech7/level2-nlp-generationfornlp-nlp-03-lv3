# fmt: off
import os
import torch
import random
import numpy as np
import logging
import logging.config
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from huggingface_hub import login
import mlflow
import mlflow.transformers

# fmt: on
hf_token = "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"
login(hf_token)

logger = logging.getLogger("gen")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

seed = 104
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정
torch.manual_seed(seed)  # torch random seed 고정
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    # 모델 불러오기
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="beomi/Solar-Ko-Recovery-11B",
        max_seq_length=4096,  # 4096
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=104,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    # 데이터 불러오기 및 전처리
    alpaca_prompt = """다음은 작업을 설명하는 지시문과 추가적인 맥락을 제공하는 입력이 함께 제공됩니다. 단계별로 추론하며 요청을 적절히 완료하는 응답을 작성하세요.
    ### 지시문:
    {}
    
    ## 입력:
    {}
    
    ## 추론:
    {}
    
    ## 추론 답변:
    {}
    
    ### 응답:
    {}"""
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        instructions = examples["Instruction"]
        inputs = examples["text"]
        resonings = examples["Reasoning"]
        resoning_answers = examples["Reasoning Answer"]
        outputs = examples["Final Answer"]
        texts = []
        for instruction, ip, r, ra, output in zip(instructions, inputs, resonings, resoning_answers, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, ip, r, ra, output) + EOS_TOKEN
            texts.append(text)

        return {"text": texts}

    dataset = load_dataset("beomi/kowikitext-qa-ref-detail-preview", split="train")
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=8,
    )
    print(dataset)
    # Trainer 초기화
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=8192,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=8,
            num_train_epochs=1,  # Set this for 1 full training run.
            learning_rate=5e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=100,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=104,
            output_dir="resources/checkpoint/knowledge",
        ),
    )
    mlflow.set_tracking_uri("http://10.28.224.137:30597/")

    # experiment를 active하고 experiment instance를 반환.
    # 원하는 실험 이름으로 바꾸기.
    mlflow.set_experiment("noah")
    # MLflow autolog 활성화
    mlflow.transformers.autolog()
    
    # Training
    with mlflow.start_run(run_name="noah_test"):  # 실험 안 run name
        trainer_stats = trainer.train()
        model.save_pretrained("kowikitext-Solar-Ko-Recovery-11B")
        tokenizer.save_pretrained("kowikitext-Solar-Ko-Recovery-11B")

        # 모델 레지스트리에 등록
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-generation",
            registered_model_name="noah_exp",  # 원하는 실험 이름으로 바꾸기.
        )

if __name__ == "__main__":
    main()
