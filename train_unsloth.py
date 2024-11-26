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
from utils.arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
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

CHAT_TEMPLETE = {
    "beomi/gemma-ko-2b": BASELINE_CHAT_TEMPLETE,
    "ludobico/gemma2_9b_it_1ep_kowiki": BASELINE_CHAT_TEMPLETE,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_CHAT_TEMPLETE,
    "hungun/Qwen2.5-14B-Instruct-kowiki-qa": QWEN_CHAT_TEMPLETE,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_CHAT_TEMPLETE,
    "MLP-KTLim/llama-3-Korean-Bllossom-8B": LLAMA3_CHAT_TEMPLETE,
    "lcw99/llama-3-10b-wiki-240709-f": LLAMA3_CHAT_TEMPLETE,
}
CHAT_TEMPLETE_EXP = {
    "hungun/Qwen2.5-14B-Instruct-kowiki-qa": QWEN_CHAT_TEMPLETE_EXP,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_CHAT_TEMPLETE_EXP,
}
CHAT_TEMPLETE_PLUS = {
    "beomi/gemma-ko-2b": BASELINE_CHAT_TEMPLETE_PLUS,
    "ludobico/gemma2_9b_it_1ep_kowiki": BASELINE_CHAT_TEMPLETE_PLUS,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_CHAT_TEMPLETE_PLUS,
    "hungun/Qwen2.5-14B-Instruct-kowiki-qa": QWEN_CHAT_TEMPLETE_PLUS,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_CHAT_TEMPLETE_PLUS,
    "MLP-KTLim/llama-3-Korean-Bllossom-8B": LLAMA3_CHAT_TEMPLETE_PLUS,
    "lcw99/llama-3-10b-wiki-240709-f": LLAMA3_CHAT_TEMPLETE_PLUS,
}
RESPONSE_TEMP = {
    "beomi/gemma-ko-2b": BASELINE_RESPONSE_TEMP,
    "ludobico/gemma2_9b_it_1ep_kowiki": BASELINE_RESPONSE_TEMP,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_RESPONSE_TEMP,
    "hungun/Qwen2.5-14B-Instruct-kowiki-qa": QWEN_RESPONSE_TEMP,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_RESPONSE_TEMP,
    "MLP-KTLim/llama-3-Korean-Bllossom-8B": LLAMA3_RESPONSE_TEMP,
    "lcw99/llama-3-10b-wiki-240709-f": LLAMA3_RESPONSE_TEMP,
}
END_TURN = {
    "beomi/gemma-ko-2b": BASELINE_END_TURN,
    "ludobico/gemma2_9b_it_1ep_kowiki": BASELINE_END_TURN,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_END_TURN,
    "hungun/Qwen2.5-14B-Instruct-kowiki-qa": QWEN_END_TURN,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_END_TURN,
    "MLP-KTLim/llama-3-Korean-Bllossom-8B": LLAMA3_END_TURN,
    "lcw99/llama-3-10b-wiki-240709-f": LLAMA3_END_TURN,
}


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
    dm = CausalLMDataModule(
        data_args,
        tokenizer,
        CHAT_TEMPLETE[model_args.model_name_or_path],
        CHAT_TEMPLETE_PLUS[model_args.model_name_or_path],
        CHAT_TEMPLETE_EXP[model_args.model_name_or_path]
    )

    train_dataset, eval_dataset = dm.get_processing_data()

    logger.info(f"{tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=False)}")
    logger.info(f"{tokenizer.decode(train_dataset[-1]['input_ids'], skip_special_tokens=False)}")
    train_dataset_token_lengths = [len(train_dataset[i]["input_ids"]) for i in range(len(train_dataset))]
    logger.info(f"max token length: {max(train_dataset_token_lengths)}")
    logger.info(f"min token length: {min(train_dataset_token_lengths)}")
    logger.info(f"avg token length: {np.mean(train_dataset_token_lengths)}")

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
    # Data collactor 설정
    logger.info(f"response template : {RESPONSE_TEMP[model_args.model_name_or_path]}")
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMP[model_args.model_name_or_path],
        tokenizer=tokenizer,
    )

    # Custom metric 설정
    logger.info(f"end turn : {END_TURN[model_args.model_name_or_path]}")
    cm = CasualMetric(tokenizer=tokenizer, end_turn=END_TURN[model_args.model_name_or_path])

    # Trainer 초기화
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    logger.info(f"토크나이저 스페셜 토큰 : {tokenizer.special_tokens_map}")

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

    mlflow.set_tracking_uri("http://10.28.224.137:30597/")

    # experiment를 active하고 experiment instance를 반환.
    # 원하는 실험 이름으로 바꾸기.
    mlflow.set_experiment("Lucia")
    # MLflow autolog 활성화
    mlflow.transformers.autolog()

    # Training
    with mlflow.start_run(run_name="no_explain"):  # 실험 안 run name
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

        # # 모델 레지스트리에 등록
        # mlflow.transformers.log_model(
        #     transformers_model={"model": trainer.model, "tokenizer": tokenizer},
        #     artifact_path="model",
        #     task="text-generation",
        #     registered_model_name="sunghoon",  # 원하는 실험 이름으로 바꾸기.
        # )


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