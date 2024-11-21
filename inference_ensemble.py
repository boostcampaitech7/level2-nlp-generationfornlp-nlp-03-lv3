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
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login

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
}
CHAT_TEMPLETE_PLUS = {
    "beomi/gemma-ko-2b": BASELINE_CHAT_TEMPLETE_PLUS,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_CHAT_TEMPLETE_PLUS,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_CHAT_TEMPLETE_PLUS,
}
RESPONSE_TEMP = {
    "beomi/gemma-ko-2b": BASELINE_RESPONSE_TEMP,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_RESPONSE_TEMP,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_RESPONSE_TEMP,
}
END_TURN = {
    "beomi/gemma-ko-2b": BASELINE_END_TURN,
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": EXAONE_END_TURN,
    "beomi/Qwen2.5-7B-Instruct-kowiki-qa-context": QWEN_END_TURN,
}

checkpoints = [
    "resources/checkpoint/fold/0/checkpoint-6468",
    "resources/checkpoint/fold/1/checkpoint-6471",
    "resources/checkpoint/fold/2/checkpoint-6471",
    "resources/checkpoint/fold/3/checkpoint-6471",
    "resources/checkpoint/fold/4/checkpoint-6471",
]

def inference_by_softvoting(checkpoints, dataset, raw_dataset, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    
    all_predictions = np.zeros((len(dataset), 5), dtype=np.float32)
    
    for model_idx, checkpoint in enumerate(checkpoints):
        print(f"현재 모델 : {model_idx+1}/{len(checkpoints)}")
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
     
        with torch.inference_mode():
            for i in tqdm(range(len(dataset)), desc=f"Model {model_idx+1} inference"):
                data = dataset[i]
                #_id = raw_dataset[i]["id"]
                len_choices = len(raw_dataset[i]["choices"].split("\n"))
                input_ids = torch.tensor(data["input_ids"], device=device)
                input_ids = input_ids.unsqueeze(0)
                #input_ids = input_ids.to(device)

                outputs = model(input_ids)
                logits = outputs.logits[:, -1].flatten().cpu()
                target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]
                probs = (
                    torch.nn.functional.softmax(torch.tensor(target_logit_list, dtype=torch.float32)).detach().cpu().numpy()
                )
                all_predictions[i] += probs
                
        del model
        torch.cuda.empty_cache()
        
    infer_results = []
    for i in range(len(dataset)):
        _id = raw_dataset[i]["id"]
        avg_probs = all_predictions[i] /len(checkpoints)
        predict_value = pred_choices_map[np.argmax(avg_probs)]
        infer_results.append({"id": _id, "answer": predict_value})
        
                    

    pd.DataFrame(infer_results).to_csv("output/output_kfold.csv", index=False)

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="logit", choices=['logit', 'generation'])
    parser.add_argument("--model_name_or_path", type=str, default="beomi/Qwen2.5-7B-Instruct-kowiki-qa-context")
    parser.add_argument("--checkpoint", type=str, default="./resources/checkpoint/qwen_aug_v3/beomi/Qwen2.5-7B-Instruct-kowiki-qa-context/checkpoint-7278")
    parser.add_argument("--dataset_name", type=str, default="./resources/raw/test_reformat.csv")
    parser.add_argument("--truncation", type=bool, default=False)
    parser.add_argument("--padding", type=bool, default=False)
    # fmt: on
    args = parser.parse_args()

    # 토크나이저 불러오기
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoints[0],
        trust_remote_code=True,
    )

    # 데이터 불러오기 및 전처리
    dm = CausalLMDataModule(
        args, tokenizer, CHAT_TEMPLETE[args.model_name_or_path], CHAT_TEMPLETE_PLUS[args.model_name_or_path]
    )
    raw_dataset, inference_dataset = dm.get_inference_data(RESPONSE_TEMP[args.model_name_or_path])
    logger.info(f"{tokenizer.decode(inference_dataset[0]['input_ids'], skip_special_tokens=False)}")
    inference_dataset_token_lengths = [len(inference_dataset[i]["input_ids"]) for i in range(len(inference_dataset))]
    logger.info(f"max token length: {max(inference_dataset_token_lengths)}")
    logger.info(f"min token length: {min(inference_dataset_token_lengths)}")
    logger.info(f"avg token length: {np.mean(inference_dataset_token_lengths)}")

    if args.strategy == "logit":
        inference_by_softvoting(checkpoints, inference_dataset, raw_dataset, tokenizer)

