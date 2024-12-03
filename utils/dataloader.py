import pandas as pd
from ast import literal_eval
from datasets import Dataset


def from_processed_train_valid(args):
    df_train = pd.read_csv(args.dataset_name)
    df_train["choices"] = [
        "\n".join([f"{idx + 1} - {choice.strip()}" for idx, choice in enumerate(literal_eval(x))])
        for x in df_train["choices"]
    ]
    try:
        df_train["explain"] = df_train["explain"].fillna("no")
    except:
        df_train["explain"] = "no"
    processed_df_train = Dataset.from_pandas(df_train)

    df_valid = pd.read_csv(args.valid_dataset_name)
    df_valid["choices"] = [
        "\n".join([f"{idx + 1} - {choice.strip()}" for idx, choice in enumerate(literal_eval(x))])
        for x in df_train["choices"]
    ]
    try:
        df_valid["explain"] = df_valid["explain"].fillna("no")
    except:
        df_valid["explain"] = "no"
    processed_df_valid = Dataset.from_pandas(df_valid)
    return processed_df_train, processed_df_valid


def from_processed(dir: str):
    df = pd.read_csv(dir)
    df["choices"] = [
        "\n".join([f"{idx + 1} - {choice.strip()}" for idx, choice in enumerate(literal_eval(x))])
        for x in df["choices"]
    ]
    try:
        df["explain"] = df["explain"].fillna("no")
    except:
        df["explain"] = "no"
    processed_df = Dataset.from_pandas(df)
    return processed_df


class CausalLMDataModule:
    def __init__(self, data_args, tokenizer, chat_templete, chat_templete_plus, chat_templete_exp=None, mode="kfold"):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.chat_templete = chat_templete
        self.chat_templete_plus = chat_templete_plus
        self.chat_templete_exp = chat_templete_exp
        if mode == "kfold":
            self.train_datasets, self.valid_datasets = from_processed_train_valid(self.data_args)
        else:
            self.datasets = from_processed(self.data_args.dataset_name)

    def _tokenize(self, instance):
        paragraph = instance["paragraph"]
        question = instance["question"]
        question_plus = instance["question_plus"]
        choices = instance["choices"]
        answer = instance["answer"]
        explain = instance["explain"]

        prompts = []
        for p, q, qp, c, e, a in zip(paragraph, question, question_plus, choices, explain, answer):
            if qp:
                prompts.append(self.chat_templete_plus.format(p, q, qp, c, a))
            else:
                if e != "no":
                    prompts.append(self.chat_templete_exp.format(p, q, c, e, a))
                else:
                    prompts.append(self.chat_templete.format(p, q, c, a))

        outputs = self.tokenizer(
            prompts,
            truncation=self.data_args.truncation,
            padding=self.data_args.padding,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    def get_processing_data(self):
        train_dataset = self.train_datasets.map(
            self._tokenize,
            remove_columns=list(self.train_datasets.features),
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
        eval_dataset = self.valid_datasets.map(
            self._tokenize,
            remove_columns=list(self.valid_datasets.features),
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )

        return train_dataset, eval_dataset

    def _tokenize_inference(self, instance):
        paragraph = instance["paragraph"]
        question = instance["question"]
        question_plus = instance["question_plus"]
        choices = instance["choices"]
        answer = instance["answer"]

        prompts = []
        for p, q, qp, c, a in zip(paragraph, question, question_plus, choices, answer):
            if qp:
                _prompt = self.chat_templete_plus.format(p, q, qp, c, a)
                _prompt = _prompt.split(self.response_temp)[0]
                prompts.append(_prompt + self.response_temp + "\n")
            else:
                _prompt = self.chat_templete.format(p, q, c, a)
                _prompt = _prompt.split(self.response_temp)[0]
                prompts.append(_prompt + self.response_temp + "\n")

        outputs = self.tokenizer(
            prompts,
            truncation=self.data_args.truncation,
            padding=self.data_args.padding,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    def get_inference_data(self, response_temp):
        self.response_temp = response_temp
        inference_dataset = self.datasets.map(
            self._tokenize_inference,
            remove_columns=list(self.datasets.features),
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
        return self.datasets, inference_dataset
