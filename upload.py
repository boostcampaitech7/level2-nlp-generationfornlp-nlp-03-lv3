import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd
import logging.config
from tqdm import tqdm
from utils.prompt import *
from utils.dataloader import CausalLMDataModule
from unsloth import FastLanguageModel
from transformers import TextStreamer
from huggingface_hub import login

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
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

if __name__ == "__main__":

    model_name_or_path="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    checkpoint='eng_toefl/checkpointunsloth/Qwen2.5-32B-Instruct-bnb-4bit/checkpoint-77'
    model, tokenizer = FastLanguageModel.from_pretrained(checkpoint, dtype=None)
    # Repository 생성 & model upload
    REPO_NAME = "Dongspam/only_Toefl" # ex) 'my-bert-fine-tuned'
    
    ## Upload to Huggingface Hub
    model.push_to_hub(
        REPO_NAME, 
        use_temp_dir=True, 
        use_auth_token=hf_token
    )
    tokenizer.push_to_hub(
        REPO_NAME, 
        use_temp_dir=True, 
        use_auth_token=hf_token
    )
