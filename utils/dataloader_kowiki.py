import pandas as pd
from ast import literal_eval
from datasets import Dataset

def from_processed(dir: str):
    df = pd.read_csv(dir)
    processed_df = Dataset.from_pandas(df)
    return processed_df

class CausalLMDataModule:
    def __init__(self, data_args, tokenizer, chat_templete, mode='train'):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.chat_templete = chat_templete
        if mode == 'train':
            self.train_datasets = from_processed(self.data_args.train_dataset_name)

    def _tokenize(self, instance):
        query = instance["query"]
        paragraph = instance["text"]
        instruction = instance["Instruction"]
        reasoning = instance["Reasoning"]
        reasoning_answer = instance["Reasoning Answer"]
        final_answer = instance["Final Answer"]
        # prefix prompt에 formatting
        prompts = []
        for q, p, i, r, ra, fa in zip(query, paragraph, instruction, reasoning, reasoning_answer, final_answer):
            prompts.append(self.chat_templete.format(q, p, i, r, ra, fa))
                             
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
        train_dataset = self.train_datasets.map(
            self._tokenize,
            remove_columns=list(self.train_datasets.features),
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )

        return train_dataset

    def _tokenize_inference(self, instance):
        paragraph = instance["paragraph"]
        question = instance["question"]
        question_plus = instance["question_plus"]
        choices = instance["choices"]
        answer = instance["answer"]

        # prefix prompt에 formatting
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