# fmt: off
import os
import torch
import random
import numpy as np
import logging
import logging.config
from utils.prompt import *
from utils.dataloader import CausalLMDataModule
from utils.arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import HfArgumentParser
from huggingface_hub import login
import mlflow
import mlflow.transformers

# fmt: on
hf_token = "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"
login(hf_token)

logger = logging.getLogger("pretrain")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

CHAT_TEMPLETE = {
    "beomi/gemma-ko-2b": BASELINE_CHAT_TEMPLETE,
    "ludobico/gemma2_9b_it_1ep_kowiki": BASELINE_CHAT_TEMPLETE,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_CHAT_TEMPLETE,
    "hungun/Qwen2.5-14B-Instruct-kowiki-qa": QWEN_CHAT_TEMPLETE,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_CHAT_TEMPLETE,
    "MLP-KTLim/llama-3-Korean-Bllossom-8B": LLAMA3_CHAT_TEMPLETE,
    "lcw99/llama-3-10b-wiki-240709-f": LLAMA3_CHAT_TEMPLETE,
    "Dongspam/toefl_and_sat": QWEN_CHAT_TEMPLETE,
}
CHAT_TEMPLETE_EXP = {
    "hungun/Qwen2.5-14B-Instruct-kowiki-qa": QWEN_CHAT_TEMPLETE_EXP,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_CHAT_TEMPLETE_EXP,
    "Dongspam/toefl_and_sat": QWEN_CHAT_TEMPLETE_EXP,
}
CHAT_TEMPLETE_PLUS = {
    "beomi/gemma-ko-2b": BASELINE_CHAT_TEMPLETE_PLUS,
    "ludobico/gemma2_9b_it_1ep_kowiki": BASELINE_CHAT_TEMPLETE_PLUS,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_CHAT_TEMPLETE_PLUS,
    "hungun/Qwen2.5-14B-Instruct-kowiki-qa": QWEN_CHAT_TEMPLETE_PLUS,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_CHAT_TEMPLETE_PLUS,
    "MLP-KTLim/llama-3-Korean-Bllossom-8B": LLAMA3_CHAT_TEMPLETE_PLUS,
    "lcw99/llama-3-10b-wiki-240709-f": LLAMA3_CHAT_TEMPLETE_PLUS,
    "Dongspam/toefl_and_sat": QWEN_CHAT_TEMPLETE_PLUS,
}
RESPONSE_TEMP = {
    "beomi/gemma-ko-2b": BASELINE_RESPONSE_TEMP,
    "ludobico/gemma2_9b_it_1ep_kowiki": BASELINE_RESPONSE_TEMP,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_RESPONSE_TEMP,
    "hungun/Qwen2.5-14B-Instruct-kowiki-qa": QWEN_RESPONSE_TEMP,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_RESPONSE_TEMP,
    "MLP-KTLim/llama-3-Korean-Bllossom-8B": LLAMA3_RESPONSE_TEMP,
    "lcw99/llama-3-10b-wiki-240709-f": LLAMA3_RESPONSE_TEMP,
    "Dongspam/toefl_and_sat": QWEN_RESPONSE_TEMP,
}
END_TURN = {
    "beomi/gemma-ko-2b": BASELINE_END_TURN,
    "ludobico/gemma2_9b_it_1ep_kowiki": BASELINE_END_TURN,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_END_TURN,
    "hungun/Qwen2.5-14B-Instruct-kowiki-qa": QWEN_END_TURN,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_END_TURN,
    "MLP-KTLim/llama-3-Korean-Bllossom-8B": LLAMA3_END_TURN,
    "lcw99/llama-3-10b-wiki-240709-f": LLAMA3_END_TURN,
    "Dongspam/toefl_and_sat": QWEN_END_TURN,
}

# Seed 설정
seed = 104
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 출력 디렉토리 설정 수정
    training_args.output_dir = training_args.output_dir + "train" + model_args.model_name_or_path.split('/')[-1]
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
    chat_template_exp = CHAT_TEMPLETE_EXP.get(model_args.model_name_or_path, None)
    
    dm = CausalLMDataModule(
        data_args,
        tokenizer,
        chat_templete=CHAT_TEMPLETE[model_args.model_name_or_path],
        chat_templete_plus=CHAT_TEMPLETE_PLUS[model_args.model_name_or_path],
        chat_templete_exp=chat_template_exp,
        mode='train'
    )
    train_dataset, _ = dm.get_processing_data()

    # 데이터셋 정보 로깅
    logger.info(f"{tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=False)}")
    logger.info(f"{tokenizer.decode(train_dataset[-1]['input_ids'], skip_special_tokens=False)}")
    train_dataset_token_lengths = [len(train_dataset[i]["input_ids"]) for i in range(len(train_dataset))]
    logger.info(f"max token length: {max(train_dataset_token_lengths)}")
    logger.info(f"min token length: {min(train_dataset_token_lengths)}")
    logger.info(f"avg token length: {np.mean(train_dataset_token_lengths)}")

    # LoRA 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=104,
        use_rslora=False,
        loftq_config=None,
    )

    # 토크나이저 설정
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    # Trainer 초기화
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
    )

    # MLflow 설정
    mlflow.set_tracking_uri("http://10.28.224.137:30597/")
    mlflow.set_experiment("merge_duplicated")
    mlflow.transformers.autolog()

    # Training
    with mlflow.start_run(run_name=f"final"):
        mlflow.log_param("lora_r", model_args.lora_r)
        mlflow.log_param("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        mlflow.log_param("lora_alpha", model_args.lora_alpha)
        mlflow.log_param("lora_dropout", 0)
        mlflow.log_param("bias", "none")
        mlflow.log_param("random_state", 104)

        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # 학습 결과 저장
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

        # # Evaluation
        # logger.info("***** Evaluate *****")
        # metrics = trainer.evaluate()
        # metrics["eval_samples"] = len(eval_dataset)
        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()