import sys
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

sys.path.append("../")
from model.dpr import Pooler
from model.bm25 import BM25Reranker
from transformers import AutoModel, AutoTokenizer
from utils.clean import WikipediaTextPreprocessor
from database.vector_database import VectorDatabase


def from_processed(dir: str):
    df = pd.read_csv(dir)
    df["choices_"] = [
        "\n".join([f"{choice.strip()}" for idx, choice in enumerate(literal_eval(x))]) for x in df["choices"]
    ]
    return df


# def inference(
#     question,
#     paragraph,
#     source,
#     q_encoder,
#     c_encoder,
#     tokenizer,
#     faiss_index,
#     search_k=2000,
#     bm25_model=None,
#     faiss_weight=1,
#     bm25_weight=0.5,
#     max_length=512,
#     pooler=None,
#     truncation=True,
#     device="cuda",
# ):
#     if source == "KMMLU":
#         c_encoder = c_encoder.to(device)
#         features = tokenizer(paragraph, max_length=max_length, truncation=truncation, return_tensors="pt").to(device)

#         c_encoder.eval()
#         with torch.no_grad():
#             c_output = c_encoder(**features, return_dict=True)

#         pooler_output = pooler(features["attention_mask"], c_output)
#         pooler_output = pooler_output.cpu().detach().numpy()  # (1, 768)

#         D, I = faiss_index.search(pooler_output, search_k)

#         if bm25_model:
#             bm25_scores = bm25_model.get_bm25_rerank_scores(paragraph, I)
#             total_scores = faiss_weight * D + bm25_weight * bm25_scores

#             sorted_idx = np.argsort(total_scores[0])[::-1]
#             D[0] = D[0][sorted_idx]
#             I[0] = I[0][sorted_idx]

#         return D, I
#     else:
#         q_encoder = q_encoder.to(device)
#         c_encoder = c_encoder.to(device)
#         q_features = tokenizer(question, max_length=max_length, truncation=truncation, return_tensors="pt").to(device)
#         c_features = tokenizer(paragraph, max_length=max_length, truncation=truncation, return_tensors="pt").to(device)

#         q_encoder.eval()
#         c_encoder.eval()
#         with torch.no_grad():
#             q_output = q_encoder(**q_features, return_dict=True)
#             c_output = c_encoder(**c_features, return_dict=True)

#         pooler_output_q = pooler(q_features["attention_mask"], q_output)
#         pooler_output_q = pooler_output_q.cpu().detach().numpy()  # (1, 768)

#         pooler_output_c = pooler(c_features["attention_mask"], c_output)
#         pooler_output_c = pooler_output_c.cpu().detach().numpy()  # (1, 768)

#         combined_pooler_output = np.concatenate((pooler_output_q, pooler_output_c), axis=0)

#         D, I = faiss_index.search(combined_pooler_output, search_k)

#         if bm25_model:
#             bm25_scores = bm25_model.get_bm25_rerank_scores(paragraph, I)
#             total_scores = faiss_weight * D + bm25_weight * bm25_scores

#             sorted_idx = np.argsort(total_scores[0])[::-1]
#             D[0] = D[0][sorted_idx]
#             I[0] = I[0][sorted_idx]

#         return D, I


def inference_batch(
    df,
    q_encoder,
    tokenizer,
    faiss_index,
    search_k=2000,
    bm25_model=None,
    faiss_weight=1,
    bm25_weight=0.5,
    max_length=512,
    pooler=None,
    truncation=True,
    device="cuda",
):
    q_encoder = q_encoder.to(device)
    q_encoder.eval()
    question_paragraph_lst = []
    question_paragraph_embed = []
    for start_index in tqdm(range(0, len(df), 64)):
        batch_row = df.iloc[start_index : start_index + 64, :]
        question = batch_row["question"].tolist()
        choices_ = batch_row["choices_"].tolist()
        paragraph = batch_row["paragraph"].tolist()
        question_paragraph = [
            q.strip() + "\n" + c.strip() + "\n" + p.strip() for q, c, p in zip(question, choices_, paragraph)
        ]
        question_paragraph_lst.extend(question_paragraph)
        features = tokenizer(
            question_paragraph, max_length=max_length, padding=True, truncation=truncation, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            c_output = q_encoder(**features, return_dict=True)

        pooler_output = pooler(features["attention_mask"], c_output).cpu()
        question_paragraph_embed.append(pooler_output)
    print(len(question_paragraph_lst), question_paragraph_lst[:2])
    question_paragraph_embed = np.vstack(question_paragraph_embed)
    print(">>> Search document by faiss")
    D, I = faiss_index.search(question_paragraph_embed, search_k)

    print(">>> Reranking")
    if bm25_model:
        bm25_scores = bm25_model.get_bm25_rerank_scores(question_paragraph_lst, I)
        total_scores = faiss_weight * D + bm25_weight * bm25_scores
        for idx in range(total_scores.shape[0]):
            sorted_idx = np.argsort(total_scores[idx])[::-1]
            I[idx] = I[idx][sorted_idx]
    return I


def main(args):
    # Load & preprocessing data
    df = from_processed(args.data)
    df = df.drop("Unnamed: 0.1", axis=1)
    df = df.drop("Unnamed: 0", axis=1)
    print(df)
    preprocessor = WikipediaTextPreprocessor()

    # Load model & tokenizer
    q_encoder = AutoModel.from_pretrained(args.question_model)
    # c_encoder = AutoModel.from_pretrained(args.context_model)
    tokenizer = AutoTokenizer.from_pretrained(args.question_model)

    pooler = Pooler(args.pooler)

    # Load faiss index.
    faiss_vector = VectorDatabase(args.faiss_path, args.context_path)
    text = faiss_vector.text
    title = faiss_vector.title
    faiss_index = faiss_vector.faiss_index

    # Load bm25 model.
    if args.bm25_path:
        bm25_model = BM25Reranker(bm25_pickle=args.bm25_path)
    else:
        bm25_model = None

    df_non_target = df[~df["source"].isin(["KMMLU", "MMMLU"])]
    print(df_non_target)
    df_target = df[df["source"].isin(["KMMLU", "MMMLU"])]
    print(df_target)

    # Retrieval loop.
    print(">>> Start KMMLU retrieve")
    I = inference_batch(
        df_target,
        q_encoder,
        tokenizer,
        faiss_index,
        search_k=args.search_k,
        bm25_model=bm25_model,
        faiss_weight=args.faiss_weight,
        bm25_weight=args.bm25_weight,
        max_length=args.max_length,
        pooler=pooler,
        truncation=args.truncation,
        device=args.device,
    )
    result_k = []
    for i_instance in I:
        context_lst = []
        for idx, index in enumerate(i_instance):
            corpus = title[index] + ". " + preprocessor.preprocess_pipeline(text[index].replace("\n", " "))
            context_lst.append(corpus)
            if idx + 1 == args.return_k:
                break
        context = "\n".join(context_lst)
        result_k.append(context)
    df_target["retrieve_context"] = result_k

    print(">>> Merge dataframe")
    df_non_target["retrieve_context"] = ""
    df_final = pd.concat([df_non_target, df_target])
    df_final = df_final.drop("choices_", axis=1)
    df_final.to_csv("train_reformat_with_source_subject_retrieve.csv", encoding="utf-8-sig", index=0)


""" 1개씩 retreival 하는 코드
    for i, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        paragraph = row["paragraph"]
        source = row["source"]
        if source == "KLUE-MRC":
            result.append("")
            continue

        D, I = inference(
            question,
            paragraph,
            source,
            q_encoder,
            c_encoder,
            tokenizer,
            faiss_index,
            search_k=args.search_k,
            bm25_model=bm25_model,
            faiss_weight=args.faiss_weight,
            bm25_weight=args.bm25_weight,
            max_length=args.max_length,
            pooler=pooler,
            truncation=args.truncation,
            device=args.device,
        )

        context_lst = []
        for idx, (distance, index) in enumerate(zip(D[0], I[0])):
            corpus = title[index] + ". " + preprocessor.preprocess_pipeline(text[index].replace("\n", " "))
            context_lst.append(corpus)
            if idx + 1 == args.return_k:
                break

        context = "\n".join(context_lst)
        result.append(context)
    df["retrieve_context"] = result
    df.to_csv("train_reformat_with_source_subject_retrieve", encoding="utf-8-sig", index=0)
"""


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="get topk-accuracy of retrieval model")
    parser.add_argument("--data", type=str, default="../resources/processed/train_reformat_with_source_subject.csv", help="Directory of pretrained encoder model")
    parser.add_argument("--question_model", type=str, default="snumin44/biencoder-ko-bert-question", help="Directory of pretrained encoder model")
    # parser.add_argument("--context_model", type=str, default="snumin44/biencoder-ko-bert-context", help="Directory of pretrained encoder model")
    parser.add_argument("--faiss_path", type=str, default="./pickles/faiss_pickle.pkl", help="Path of faiss pickle")
    parser.add_argument("--context_path", type=str, default="./pickles/context_pickle.pkl", help="Path of context pickle")
    parser.add_argument("--bm25_path", type=str, default="./pickles/bm25_pickle.pkl", help="Path of BM25 Model")
    parser.add_argument("--faiss_weight", default=0.8, type=float, help="Weight for semantic search")
    parser.add_argument("--bm25_weight", default=0.2, type=float, help="Weight for BM25 rerank score")
    parser.add_argument("--search_k", default=2000, type=int, help="Number of retrieved documents")
    parser.add_argument("--return_k", default=10, type=int, help="Number of returned documents")
    parser.add_argument("--max_length", default=512, type=int, help="Max length of sequence")
    parser.add_argument("--pooler", default="cls", type=str, help="Pooler type : {pooler_output|cls|mean|max}")
    parser.add_argument("--truncation", action="store_false", default=True, help="Truncate extra tokens when exceeding the max_length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Choose a type of device for training")
    args = parser.parse_args()
    # fmt: on

    main(args)
