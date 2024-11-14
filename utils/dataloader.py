import pandas as pd
from ast import literal_eval
from datasets import Dataset, DatasetDict


def from_processed(dir: str):
    df = pd.read_csv(dir)
    df["choices"] = [
        "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(literal_eval(x))]) for x in df["choices"]
    ]
    df["question_plus"] = df["question_plus"].fillna("")
    processed_df = Dataset.from_pandas(df)
    return processed_df


class CausalLMDataModule:
    def __init__(self, data_args, tokenizer, chat_templete):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.chat_templete = chat_templete
        self.datasets = from_processed(data_args.dataset_name)

    def _tokenize(self, instance):
        paragraph = instance["paragraph"]  # dataset batch에서 question 가져오기
        question = instance["question"]  # dataset batch에서 paragraph 가져오기
        choices = instance["choices"]  # dataset batch에서 answer 가져오기
        answer = instance["answer"]  # dataset batch에서 answer 가져오기

        # prefix prompt에 formatting
        prompts = [self.chat_templete.format(p, q, c, a) for p, q, c, a in zip(paragraph, question, choices, answer)]

        # tokenization
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
        tokenized_dataset = self.datasets.map(
            self._tokenize,
            remove_columns=list(self.datasets.features),
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=104)
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= 1024)
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["test"]
        return train_dataset, eval_dataset
