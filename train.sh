#!/bin/bash

# Python 환경 설정
source .venv/bin/activate

# 현재 디렉토리 출력
echo "현재 작업 디렉토리: $(pwd)"

# train.py 실행
echo "train.py 실행 중..."
python train_unsloth.py --model_name_or_path hungun/Qwen2.5-14B-Instruct-kowiki-qa
python train_unsloth.py --model_name_or_path ludobico/gemma2_9b_it_1ep_kowiki
python train_unsloth.py --model_name_or_path /dev/shm/lcw99/llama-3-10b-wiki-240709-f
# python train_unsloth.py --model_name_or_path MLP-KTLim/llama-3-Korean-Bllossom-8B

# 실행 완료 메시지
echo "train.py 실행 완료"