# fmt: off
import os
import torch
import random
import numpy as np
import logging
import logging.config
from utils.prompt import *
from utils.metric import CasualMetric
from utils.helpers import find_linear_names
from utils.dataloader import CausalLMDataModule
from utils.arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

CHAT_TEMPLETE = {
    "beomi/gemma-ko-2b": BASELINE_CHAT_TEMPLETE,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_CHAT_TEMPLETE,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_CHAT_TEMPLETE,
    "Qwen/Qwen2.5-14B-Instruct": QWEN_CHAT_TEMPLETE,
}
CHAT_TEMPLETE_PLUS = {
    "beomi/gemma-ko-2b": BASELINE_CHAT_TEMPLETE_PLUS,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_CHAT_TEMPLETE_PLUS,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_CHAT_TEMPLETE_PLUS,
    "Qwen/Qwen2.5-14B-Instruct": QWEN_CHAT_TEMPLETE_PLUS,
}
RESPONSE_TEMP = {
    "beomi/gemma-ko-2b": BASELINE_RESPONSE_TEMP,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_RESPONSE_TEMP,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_RESPONSE_TEMP,
    "Qwen/Qwen2.5-14B-Instruct": QWEN_RESPONSE_TEMP,
}
END_TURN = {
    "beomi/gemma-ko-2b": BASELINE_END_TURN,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_END_TURN,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_END_TURN,
    "Qwen/Qwen2.5-14B-Instruct": QWEN_END_TURN,
}


def main():
    torch.cuda.empty_cache()
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)  # arguement 쭉 읽어보면서 이해하기
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = training_args.output_dir + model_args.model_name_or_path

    # 토크나이저 불러오기
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )

    # 데이터 불러오기 및 전처리
    dm = CausalLMDataModule(
        data_args,
        tokenizer,
        CHAT_TEMPLETE[model_args.model_name_or_path],
        CHAT_TEMPLETE_PLUS[model_args.model_name_or_path],
    )
    train_dataset, eval_dataset = dm.get_processing_data()
    logger.info(f"{tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=False)}")
    logger.info(f"{tokenizer.decode(train_dataset[1]['input_ids'], skip_special_tokens=False)}")
    logger.info(f"{tokenizer.decode(train_dataset[2]['input_ids'], skip_special_tokens=False)}")
    train_dataset_token_lengths = [len(train_dataset[i]["input_ids"]) for i in range(len(train_dataset))]
    logger.info(f"max token length: {max(train_dataset_token_lengths)}")
    logger.info(f"min token length: {min(train_dataset_token_lengths)}")
    logger.info(f"avg token length: {np.mean(train_dataset_token_lengths)}")

    # QLoRA 사용할지
    if model_args.quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    # LoRA 설정 : 모든 모듈 쓰면 터져서 일부 모듈만 사용
    if model_args.quantization:
        modules = find_linear_names(model, "qlora")
    else:
        modules = find_linear_names(model, "lora")

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj"],
        # target_modules=modules,
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )

    # 모델 불러오기
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data collactor 설정
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMP[model_args.model_name_or_path],
        tokenizer=tokenizer,
    )

    # Custom metric 설정
    cm = CasualMetric(tokenizer=tokenizer, end_turn=END_TURN[model_args.model_name_or_path])

    # Trainer 초기화
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    logger.info(f"토크나이저 스페셜 토큰 : {tokenizer.special_tokens_map}")

    mlflow.set_tracking_uri("http://10.28.224.137:30597/")
    # experiment를 active하고 experiment instance를 반환.
    # 원하는 실험 이름으로 바꾸기.
    mlflow.set_experiment("baseline")
    # MLflow autolog 활성화
    mlflow.transformers.autolog()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=cm.compute_metrics,
        preprocess_logits_for_metrics=cm.preprocess_logits_for_metrics,
        args=training_args,
    )

    # Training
    with mlflow.start_run(run_name="exp"):  # 실험 안 run name
        mlflow.log_params(lora_config.to_dict())
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

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

        # Evaluation
        logger.info("***** Evaluate *****")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
