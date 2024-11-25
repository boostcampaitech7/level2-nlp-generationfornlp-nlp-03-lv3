import os
from typing import Optional
from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, field
from trl import SFTConfig
from datetime import datetime


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
            "baseline : beomi/gemma-ko-2b / LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct / beomi/Qwen2.5-7B-Instruct-kowiki-qa-context"
            "beomi/Solar-Ko-Recovery-11B"
            "hungun/Qwen2.5-14B-Instruct-kowiki-qa"
            "ludobico/gemma2_9b_it_1ep_kowiki"
            "MLP-KTLim/llama-3-Korean-Bllossom-8B"
            "lcw99/llama-3-10b-wiki-240709-f"
        },
    )
    quantization: bool = field(
        default=False,
        metadata={"help": "QLoRA(4bit) 사용할지 안할지, 만약 사용한다면 optim 수정, 대신 학습 속도가 느려짐"},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "학습 할 에폭 수" "LLM 학습 시 에폭 수를 1~3으로 줄여서 실험 진행 필요"},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "학습 할 에폭 수" "LLM 학습 시 에폭 수를 1~3으로 줄여서 실험 진행 필요"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # 학습 데이터 불러오기
    dataset_name: str = field(
        default="./resources/merge/merge_dataset_20241125.csv",
        metadata={"help": "The name of the dataset to use."},
    )
    # 검증 데이터 불러오기
    dataset_name: str = field(
        default="./resources/merge/merge_dataset_20241125.csv",
        metadata={"help": "The name of the dataset to use."},
    )
    # 토크나이저 설정
    truncation: bool = field(
        default=False,
        metadata={
            "help": "입력 텍스트가 모델의 최대 허용 길이를 초과하더라도 잘리지 않고 그대로 유지"
            "EXAONE maximum context length : 4096"
        },
    )
    padding: bool = field(
        default=False,
        metadata={
            "help": "DataCollatorForCompletionOnlyLM 을 통해 배치 내에서 가장 긴 시퀀스의 길이에 맞춰 다른 시퀀스들을 패딩"
        },
    )


@dataclass
class OurTrainingArguments(SFTConfig):
    """
    HuggingFace의 transformers 라이브러리에서 모델 학습할때 사용되는 하이퍼파라미터 커스텀
    """

    # 기본 학습 설정
    output_dir: Optional[str] = field(
        default="./resources/checkpoint/",
        metadata={"help": "체크포인트와 모델 출력을 저장할 디렉터리 경로"},
    )
    max_seq_length: int = field(
        default=3000,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "학습을 실행할지 여부"},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "평가를 실행할지 여부"},
    )
    # 학습 관련 설정
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "학습 할 에폭 수" "LLM 학습 시 에폭 수를 1~3으로 줄여서 실험 진행 필요"},
    )
    # max_steps: int = field(
    #     default=3,
    #     metadata={
    #         "help": "학습 할 스텝 수"
    #     },
    # )
    eval_strategy: Optional[str] = field(
        default="epoch",
        metadata={"help": "epoch/steps이 끝날때마다 평가"},
    )
    # save_steps: int = field(
    #     default=200,
    #     metadata={"help": "어떤 step에서 저장할지"},
    # )
    # eval_steps: int = field(
    #     default=5,
    #     metadata={"help": "어떤 step에서 저장할지"},
    # )
    logging_steps: int = field(default=200)
    save_strategy: Optional[str] = field(
        default="epoch",
        metadata={"help": "epoch/steps이 끝날때마다 저장"},
    )
    save_total_limit: int = field(
        default=None,
        metadata={"help": "가장 좋은 체크포인트 n개만 저장하여 이전 모델을 덮어씌우도록 설정"},
    )
    save_only_model: bool = field(default=False)
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "가장 좋은 모델 로드"},
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "학습 중 장치당 배치 크기" "GPU 메모리에 따라 줄여서 사용 / 너무 큰 배치는 지양"},
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={
            "help": "평가 중 장치당 배치 크기"
            "per_device_eval_batch_size 따라 accuracy 값이 다르게 나옴"
            "지정된 batch 내에서 accuracy를 계산해서 그런 것 같은데 근데 1일 때는 어떻게 계산하는 지 모르겠음"
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "그래디언트 누적을 위한 스텝 수" "GPU 자원이 부족할 시 배치를 줄이고 누적 수를 늘려 학습"},
    )
    learning_rate: int = field(
        default=2e-05,
        metadata={"help": "학습률 설정" "학습률 스케줄러(linear, cosine) 사용시 Max 값"},
    )
    # 모델 평가 관련
    metric_for_best_model: Optional[str] = field(
        default="eval_loss",
        metadata={"help": "가장 좋은 모델을 평가하기 위한 메트릭 설정" "본 프로젝트에서는 eval_loss를 기본적으로 사용"},
    )
    greater_is_better: bool = field(
        default=False,
        metadata={
            "help": "설정한 메트릭에 대해 더 큰 값이 더 좋다 혹은 더 작은 값이 더 좋다 설정"
            "Accuracy는 True 사용 / eval_loss는 False 사용"
        },
    )
    # Optimizer 설정
    optim: str = field(
        default="adamw_8bit",
        metadata={
            "help": "옵티마이저 설정, 다른 옵티마이저 확인을 위해 아래 url에서 OptimizerNames 확인"
            "Default : adamw_torch / QLoRA 사용시 : paged_adamw_8bit / adamw_8bit"
            "https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py"
        },
    )
    weight_decay: int = field(
        default=0.01,
        metadata={"help": "가중치 감소율 (정규화), 과적합 방지" "0.01 ~ 0.1 정도가 많이 사용"},
    )
    max_grad_norm: int = field(
        default=1,
        metadata={
            "help": "그래디언트 클리핑을 위한 최대 노름"
            "1 또는 그 이상의 값으로 설정하는 것이 일반적, 하지만 때에 따라(예를들어 LLM SFT시) 0도 설정 해보길 권장"
        },
    )
    # 스케줄러 설정
    lr_scheduler_type: Optional[str] = field(
        default="cosine",  # cosine_with_restarts
        metadata={"help": "학습률 스케줄러 설정" "cosine_with_restarts"},
    )
    # warmup_steps: int = field(
    #     default=0,
    #     metadata={
    #         "help": "학습률을 워밍업하기 위한 스텝 수"
    #         "전체 학습 스텝 수의 2%~5% 정도로 설정하는 것이 일반적"
    #         "스텝수 = 데이터 개수*에폭수 / 배치사이즈"
    #     },
    # )
    report_to: Optional[str] = field(
        default="mlflow",
        metadata={"help": "mlflow로 logging"},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("training_args : ", training_args)
