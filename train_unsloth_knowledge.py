# fmt: off
import os
import torch
import random
import numpy as np
import logging
import logging.config
from utils.prompt import *
from utils.metric import CasualMetric
from utils.dataloader import CausalLMDataModule
from utils.arguments_knowledge import ModelArguments, DataTrainingArguments, OurTrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from datasets import load_from_disk
from transformers import HfArgumentParser
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
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)  # arguement 쭉 읽어보면서 이해하기
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = training_args.output_dir + model_args.model_name_or_path
    training_args.fp16 = not is_bfloat16_supported()
    training_args.bf16 = is_bfloat16_supported()

    # 모델 및 토크나이저 불러오기
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        dtype=None,
        load_in_4bit=model_args.quantization,
    )
    logger.info(f">>> {model_args.model_name_or_path}")

    # 데이터 불러오기 및 전처리
    dataset = load_from_disk('./resources/selected_koreanTextbook')
    text = []
    for x in dataset:
        text.append(x['text'])

    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        # target_modules=["q_proj", "k_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=104,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    # Trainer 초기화
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
    )

    mlflow.set_tracking_uri("http://10.28.224.137:30597/")

    # experiment를 active하고 experiment instance를 반환.
    # 원하는 실험 이름으로 바꾸기.
    mlflow.set_experiment("textbook")
    # MLflow autolog 활성화
    mlflow.transformers.autolog()

    # Training
    with mlflow.start_run(run_name="1127"):  # 실험 안 run name
        mlflow.log_param("lora_r", model_args.lora_r)
        mlflow.log_param("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        mlflow.log_param("lora_alpha", model_args.lora_alpha)
        mlflow.log_param("lora_dropout", 0)
        mlflow.log_param("bias", "none")
        mlflow.log_param("lora_alpha", model_args.lora_alpha)
        mlflow.log_param("random_state", 104)

        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Training state 저장
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


if __name__ == "__main__":
    main()

"""
if model_args.model_name_or_path == "beomi/Solar-Ko-Recovery-11B":
    token_list = ["<|im_start|>", "<|im_end|>"]
    special_tokens_dict = {"additional_special_tokens": token_list}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"{special_tokens_dict}")
"""
