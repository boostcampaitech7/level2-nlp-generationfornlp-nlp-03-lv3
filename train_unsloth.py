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
from transformers import HfArgumentParser, AutoTokenizer
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
    "gemma-2-27b-bnb-4bit": BASELINE_CHAT_TEMPLETE,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_CHAT_TEMPLETE,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_CHAT_TEMPLETE_EN, #QWEN_CHAT_TEMPLETE #flag
    "LoneStriker/aya-23-35B-4.65bpw-h6-exl2": SOLAR_CHAT_TEMPLETE,
}
CHAT_TEMPLETE_PLUS = {
    "gemma-2-27b-bnb-4bit": BASELINE_CHAT_TEMPLETE_PLUS,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_CHAT_TEMPLETE_PLUS, 
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_CHAT_TEMPLETE_PLUS_EN, #QWEN_CHAT_TEMPLETE_PLUS TOEFL data has no question_plus
    "LoneStriker/aya-23-35B-4.65bpw-h6-exl2": SOLAR_CHAT_TEMPLETE_PLUS,
}
RESPONSE_TEMP = {
    "gemma-2-27b-bnb-4bit": BASELINE_RESPONSE_TEMP,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_RESPONSE_TEMP,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_RESPONSE_TEMP_EN, #QWEN_RESPONSE_TEMP
    "LoneStriker/aya-23-35B-4.65bpw-h6-exl2": SOLAR_RESPONSE_TEMP,
}
END_TURN = {
    "gemma-2-27b-bnb-4bit": BASELINE_END_TURN,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_END_TURN,
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": QWEN_END_TURN_EN, #QWEN_END_TURN
    "LoneStriker/aya-23-35B-4.65bpw-h6-exl2": SOLAR_END_TURN,
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
        dtype="None",
        load_in_4bit=True,
        use_gradient_checkpointing=True,
        offload_folder="offload/",
    )
    if model_args.model_name_or_path in ["LoneStriker/aya-23-35B-4.65bpw-h6-exl2"]:
        token_list = ["<|im_start|>", "<|im_end|>"]
        special_tokens_dict = {"additional_special_tokens": token_list}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"{special_tokens_dict}")

    #데이터 불러오기 및 전처리
    dm = CausalLMDataModule(
        data_args,
        tokenizer,
        CHAT_TEMPLETE[model_args.model_name_or_path],
        CHAT_TEMPLETE_PLUS[model_args.model_name_or_path],
    )

    train_dataset = dm.get_processing_data() #flag

    logger.info(f"{tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=False)}")
    logger.info(f"{tokenizer.decode(train_dataset[1]['input_ids'], skip_special_tokens=False)}")
    logger.info(f"{tokenizer.decode(train_dataset[2]['input_ids'], skip_special_tokens=False)}")
    train_dataset_token_lengths = [len(train_dataset[i]["input_ids"]) for i in range(len(train_dataset))]
    logger.info(f"max token length: {max(train_dataset_token_lengths)}")
    logger.info(f"min token length: {min(train_dataset_token_lengths)}")
    logger.info(f"avg token length: {np.mean(train_dataset_token_lengths)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        #target_modules=["q_proj", "k_proj", "v_proj"],
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
        # eval_dataset=eval_dataset, # flag
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=cm.compute_metrics,
        preprocess_logits_for_metrics=cm.preprocess_logits_for_metrics,
        args=training_args,
    )
    
     
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


# 바꿀 것
# 모델 이름
# 데이터셋 이름
# 실험, run, registry 이름
# 프롬프트 딕셔너리 이름 추가

if __name__ == "__main__":
    main()