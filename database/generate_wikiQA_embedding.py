import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append("../")
from model.bm25 import BM25Reranker

LOGGER = logging.getLogger()


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    init_logging()
    seed_everything(args)

    LOGGER.info("*** Building Vector Database ***")
    tokenizer = AutoTokenizer.from_pretrained("beomi/Solar-Ko-Recovery-11B")

    dataset = load_dataset("beomi/kowikitext-qa-ref-detail-preview", split="train")
    print(dataset)
    title_lst = []
    txt_lst = []
    for d in dataset:
        print(d)
        exit()

    pickle_file_path = os.path.join(args.save_path, "context_pickle.pkl")
    with open(pickle_file_path, "wb") as file:
        pickle.dump(
            {
                "title": title_lst,
                "text": txt_lst,
            },
            file,
        )
    vector_db.build_embedding(
        wiki_path=args.wiki_path,
        save_path=args.save_path,
        save_context=args.save_context,
        tokenizer=tokenizer,
        embedding_model=model,
        pooler=pooler,
        cpu_workers=args.cpu_workers,
        gold_passages=None,
        device=args.device,
    )

    #### Train BM 25 ####
    print(">>> Train BM 25")
    if args.train_bm25:
        bm25_model = BM25Reranker(tokenizer=tokenizer)
        bm25_model.build_bm25_model(text=vector_db.text, title=vector_db.title, path=args.save_path)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Build vector database with wiki text")
    parser.add_argument("--save_path", type=str, default="./pickles", help="Save directory of faiss index")
    parser.add_argument("--save_context", action="store_true", default=True, help="Save text and title with faiss index")
    parser.add_argument("--train_bm25", action="store_true", default=True, help="Train bm25 with the same corpus")
    parser.add_argument("--num_sent", type=int, default=5, help="Number of sentences consisting of a wiki chunk")
    parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences between consecutive chunks")
    parser.add_argument("--pooler", default="cls", type=str, help="Pooler type : {pooler_output|cls|mean|max}")
    parser.add_argument("--max_length", type=int, default=512, help="Max length for encoder model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--cpu_workers", type=int, default=50, required=False, help="Number of cpu cores used in chunking wiki text")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Choose a type of device for training")
    parser.add_argument("--random_seed", default=104, type=int, help="Random seed")
    # fmt: on
    args = parser.parse_args()

    main(args)
