import os
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

CHAT_TEMPLETE = {
    "beomi/gemma-ko-2b": BASELINE_CHAT_TEMPLETE,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_CHAT_TEMPLETE,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_CHAT_TEMPLETE,
    "beomi/Solar-Ko-Recovery-11B": SOLAR_CHAT_TEMPLETE,
}
CHAT_TEMPLETE_PLUS = {
    "beomi/gemma-ko-2b": BASELINE_CHAT_TEMPLETE_PLUS,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_CHAT_TEMPLETE_PLUS,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_CHAT_TEMPLETE_PLUS,
    "beomi/Solar-Ko-Recovery-11B": SOLAR_CHAT_TEMPLETE_PLUS,
}
CHAT_TEMPLETE_R = {
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": [QWEN_CHAT_TEMPLETE_R, QWEN_CHAT_TEMPLETE_PLUS_R],
}
RESPONSE_TEMP = {
    "beomi/gemma-ko-2b": BASELINE_RESPONSE_TEMP,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_RESPONSE_TEMP,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_RESPONSE_TEMP,
    "beomi/Solar-Ko-Recovery-11B": SOLAR_RESPONSE_TEMP,
}
END_TURN = {
    "beomi/gemma-ko-2b": BASELINE_END_TURN,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_END_TURN,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_END_TURN,
    "beomi/Solar-Ko-Recovery-11B": SOLAR_END_TURN,
}


def inference_by_logit(model, dataset, raw_dataset, tokenizer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    infer_results = []

    model.eval()
    with torch.inference_mode():
        for i, data in tqdm(enumerate(dataset), total=len(dataset)):
            _id = raw_dataset[i]["id"]
            len_choices = len(raw_dataset[i]["choices"].split("\n"))
            input_ids = torch.tensor(data["input_ids"])
            input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(device)

            outputs = model(input_ids)

            logits = outputs.logits[:, -1].flatten().cpu()
            target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]
            probs = (
                torch.nn.functional.softmax(torch.tensor(target_logit_list, dtype=torch.float32)).detach().cpu().numpy()
            )
            predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
            infer_results.append({"id": _id, "answer": predict_value})

    pd.DataFrame(infer_results).to_csv("output.csv", index=False)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="logit", choices=['logit', 'generation'])
    parser.add_argument("--model_name_or_path", type=str, default="beomi/Solar-Ko-Recovery-11B")
    parser.add_argument("--checkpoint", type=str, default="./resources/checkpoint/dev/shm/model/kowikitext-Solar-Ko-Recovery-11B/checkpoint-2792")
    parser.add_argument("--dataset_name", type=str, default="./resources/raw/test_reformat.csv")
    parser.add_argument("--truncation", type=bool, default=False)
    parser.add_argument("--padding", type=bool, default=False)
    # fmt: on
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(args.checkpoint, dtype=None)
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    # 데이터 불러오기 및 전처리
    dm = CausalLMDataModule(
        args,
        tokenizer,
        CHAT_TEMPLETE[args.model_name_or_path],
        CHAT_TEMPLETE_PLUS[args.model_name_or_path],
        # CHAT_TEMPLETE_R[args.model_name_or_path] if args.model_name_or_path != "beomi/Solar-Ko-Recovery-11B" else None,
    )
    raw_dataset, inference_dataset = dm.get_inference_data(RESPONSE_TEMP[args.model_name_or_path])
    logger.info(f"{tokenizer.decode(inference_dataset[0]['input_ids'], skip_special_tokens=False)}")
    logger.info(f"{tokenizer.decode(inference_dataset[6]['input_ids'], skip_special_tokens=False)}")
    logger.info(f"{tokenizer.decode(inference_dataset[-2]['input_ids'], skip_special_tokens=False)}")
    inference_dataset_token_lengths = [len(inference_dataset[i]["input_ids"]) for i in range(len(inference_dataset))]
    logger.info(f"max token length: {max(inference_dataset_token_lengths)}")
    logger.info(f"min token length: {min(inference_dataset_token_lengths)}")
    logger.info(f"avg token length: {np.mean(inference_dataset_token_lengths)}")

    if args.strategy == "logit":
        inference_by_logit(model, inference_dataset, raw_dataset, tokenizer)
