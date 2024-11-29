import sys
import argparse
import pandas as pd
from ast import literal_eval
from datasets import load_from_disk, load_dataset
from queries import QUERIES
sys.path.append("../")
from model.bm25 import BM25SModel
from transformers import AutoTokenizer


def from_processed(dir: str):
    df = pd.read_csv(dir)
    df["choices_"] = [
        "\n".join([f"{choice.strip()}" for idx, choice in enumerate(literal_eval(x))]) for x in df["choices"]
    ]
    return df


def main(args):
    # Load & preprocessing data
    train_df = from_processed(args.train_data)
    test_df = from_processed(args.test_data)
    df = pd.concat([train_df, test_df])
    print(len(df))
    queries = []
    for i, row in df.iterrows():
        question = row["question"]
        choices_ = row["choices_"]
        paragraph = row["paragraph"]
        queries.append(question * 3 + "\n" + paragraph + "\n" + choices_)
    print(len(queries))
    # queries = QUERIES
    dataset = load_dataset("maywell/korean_textbooks", "tiny-textbooks")
    # dataset = load_from_disk("../resources/koreanTextbook")
    dataset = dataset["train"]
    print(dataset)

    # Load bm25 model.
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    bm25_model = BM25SModel(tokenizer=tokenizer, bm25_dir=args.bm25_path)

    # Retrieval.
    print(">>> Get Scores")
    top_k_indices = bm25_model.get_bm25_rank_scores(queries, topk=2)
    print(">>> Check indices len")
    print(len(top_k_indices))
    top_k_indices = list(set(top_k_indices))
    print(len(top_k_indices))
    selected_data = dataset.select(top_k_indices)
    selected_data.save_to_disk("../resources/selected_koreanTextbook_1")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="get topk-accuracy of retrieval model")
    parser.add_argument("--train_data", type=str, default="../resources/raw/train_reformat.csv", help="Path of context pickle")
    parser.add_argument("--test_data", type=str, default="../resources/raw/test_reformat.csv", help="Path of context pickle")
    parser.add_argument("--bm25_path", type=str, default="./bm25_model_textbook", help="Path of BM25 Model")
    parser.add_argument("--search_k", default=2000, type=int, help="Number of retrieved documents")
    args = parser.parse_args()
    # fmt: on

    main(args)
