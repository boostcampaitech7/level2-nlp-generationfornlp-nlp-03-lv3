import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig


class MLPLayer(nn.Module):
    def __init__(self, dropout, config):
        super(MLPLayer, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        x = self.dense(features)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    def __init__(self, temp):
        super(Similarity, self).__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    def __init__(self, pooler_type):
        super(Pooler, self).__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["pooler_output", "cls", "mean", "max"]

    def forward(self, attention_mask, outputs):
        last_hidden_state = outputs.last_hidden_state
        # hidden_states = outputs.hidden_states

        if self.pooler_type == "pooler_output":
            return outputs.pooler_output

        elif self.pooler_type == "cls":
            return last_hidden_state[:, 0, :]

        # code from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py#L9-L241
        elif self.pooler_type == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask  # mean_embeddings : (batch_size, hidden_size)
            return mean_embeddings

        # code from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py#L9-L241
        elif self.pooler_type == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_embeddings = torch.max(last_hidden_state, 1)[0]  # max_embeddings : (batch_size, hidden_size)
            return max_embeddings

        else:
            raise NotImplementedError


class BiEncoder(nn.Module):
    def __init__(self, args):
        super(BiEncoder, self).__init__()
        self.config = AutoConfig.from_pretrained(args.model)
        self.q_encoder = AutoModel.from_pretrained(args.model, return_dict=True)
        self.c_encoder = AutoModel.from_pretrained(args.model, return_dict=True)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, config=self.config)
        self.q_encoder.resize_token_embeddings(len(self.tokenizer))
        self.c_encoder.resize_token_embeddings(len(self.tokenizer))

        self.q_mlp = MLPLayer(args.dropout, self.config)
        self.c_mlp = MLPLayer(args.dropout, self.config)

        self.sim = Similarity(args.temp)
        self.pooler = Pooler(args.pooler)
        self.device = args.device

    @autocast()
    def forward(self, q_input_ids, q_attn_mask, q_token_ids, c_input_ids, c_attn_mask, c_token_ids):

        # q_output_pooled : (batch_size, hidden_size)
        q_output = self.q_encoder(q_input_ids, q_attn_mask, q_token_ids)
        q_output_pooled = self.pooler(q_attn_mask, q_output)

        # c_batch : (batch size, 2, sequence_length) when there are hard negatives
        #           (batch size, 1, sequence_length) when there is no hard negative
        c_batch_size, c_ctx_type = c_input_ids.size(0), c_input_ids.size(1)

        # Flatten input features for encoding.
        # c_input_ids, c_attn_mask, c_token_ids: (batch_size * (2 or 1), sequence_length)
        c_input_ids = c_input_ids.view((-1, c_input_ids.size(-1)))
        c_attn_mask = c_attn_mask.view((-1, c_attn_mask.size(-1)))
        c_token_ids = c_token_ids.view((-1, c_token_ids.size(-1)))

        # c_output_pooled : (batch_size * (2 or 1), hidden_size)
        c_output = self.c_encoder(c_input_ids, c_attn_mask, c_token_ids)
        c_output_pooled = self.pooler(c_attn_mask, c_output)

        # c_output_pooled : (batch_size, 2 or 1, hidden_size)
        c_output_pooled = c_output_pooled.view((c_batch_size, c_ctx_type, c_output_pooled.size(-1)))

        # Seperate representations
        # c_pos_pooled, c_neg_pooled : (batch_size, hidden_size)
        c_pos_pooled = c_output_pooled[:, 0]
        if c_ctx_type == 2:
            c_neg_pooled = c_output_pooled[:, 1]

        # All outputs except the 'pooler_output' are passed through the mlp layer.
        # output_pooled = (batch_size, hidden_size)
        if self.pooler in ["cls", "mean", "max"]:
            q_output_pooled = self.q_mlp(q_output_pooled)
            c_pos_pooled = self.c_mlp(c_pos_pooled)

            if c_ctx_type == 2:
                c_neg_pooled = self.c_mlp(c_neg_pooled)

        # calculate cosine similarity
        total_sim = self.sim(q_output_pooled.unsqueeze(1), c_pos_pooled.unsqueeze(0))
        if c_ctx_type == 2:
            hard_neg_sim = self.sim(q_output_pooled.unsqueeze(1), c_neg_pooled.unsqueeze(0))
            total_sim = torch.cat([total_sim, hard_neg_sim], 1)

        labels = torch.arange(total_sim.size(0)).to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(total_sim, labels)
        return loss

    def get_q_embeddings(self, input_ids, attention_mask, token_type_ids):
        q_output = self.q_encoder(input_ids, attention_mask, token_type_ids)
        q_output_pooled = self.pooler(attention_mask, q_output)
        return q_output_pooled

    def get_c_embeddings(self, input_ids, attention_mask, token_type_ids):
        c_output = self.c_encoder(input_ids, attention_mask, token_type_ids)
        c_output_pooled = self.pooler(attention_mask, c_output)
        return c_output_pooled

    def save_model(self, q_output_path, c_output_path):
        # save encoder only.
        self.q_encoder.save_pretrained(q_output_path)
        self.c_encoder.save_pretrained(c_output_path)
        # save tokenizer.
        self.tokenizer.save_pretrained(q_output_path)
        self.tokenizer.save_pretrained(c_output_path)


class BiEncoderDataset(Dataset):
    def __init__(self, question, positive_ctx, answer_idx=None, hard_neg_ctx=None):
        super(BiEncoderDataset, self).__init__()
        self.question = question
        self.answer_idx = answer_idx
        self.positive_ctx = positive_ctx
        self.hard_neg_ctx = hard_neg_ctx

    @classmethod
    def load_train_dataset(cls, file_path):
        with open(file_path) as infile:
            dataset = json.load(infile)

        question, positive_ctx, hard_neg_ctx = [], [], []

        for idx, sample in enumerate(dataset):
            p = sample["positive"]
            q = sample["question"]

            positive_ctx.extend(p)
            question.extend([q] * len(p))

            if "hard_neg" in sample.keys():
                if len(p) == len(sample["hard_neg"]):
                    hard_neg_ctx.extend(sample["hard_neg"])

        if hard_neg_ctx and len(hard_neg_ctx) == len(positive_ctx):
            return cls(question, positive_ctx, hard_neg_ctx=hard_neg_ctx)
        else:
            if hard_neg_ctx:
                print(
                    f"The number of hard negatives ({len(hard_neg_ctx)}) does not match the number of positives ({len(positive_ctx)}). Hard negatives were not loaded."
                )

        return cls(question, positive_ctx)

    @classmethod
    def load_valid_dataset(cls, file_path):
        with open(file_path) as infile:
            dataset = json.load(infile)

        question, answer_idx, positive_ctx = [], [], []

        for idx, sample in enumerate(dataset):
            p = sample["positive"]
            q = sample["question"]
            a = sample["answer_idx"]

            positive_ctx.extend(p)
            answer_idx.extend([a])
            question.extend([q])

        return cls(question, positive_ctx, answer_idx=answer_idx)

    def __len__(self):
        return len(self.question)

    def __getitem__(self, index):
        if self.hard_neg_ctx:
            return {
                "question": self.question[index],
                "positive_ctx": self.positive_ctx[index],
                "hard_neg_ctx": self.hard_neg_ctx[index],
            }
        else:
            return {
                "question": self.question[index],
                "positive_ctx": self.positive_ctx[index],
            }


class DataCollator(object):

    def __init__(self, args):
        self.config = AutoConfig.from_pretrained(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, config=self.config)

        self.padding = args.padding
        self.max_length = args.max_length
        self.truncation = args.truncation

    def __call__(self, samples):
        question = []
        positive_idx, positive_txt, positive_title = [], [], []
        hard_neg_idx, hard_neg_txt, hard_neg_title = [], [], []

        for sample in samples:
            question.append(sample["question"])
            positive_idx.append(sample["positive_ctx"]["idx"])
            positive_txt.append(sample["positive_ctx"]["text"])
            positive_title.append(sample["positive_ctx"]["title"])

            if "hard_neg_ctx" in sample.keys():
                hard_neg_idx.append(sample["hard_neg_ctx"]["idx"])
                hard_neg_txt.append(sample["hard_neg_ctx"]["text"])
                hard_neg_title.append(sample["hard_neg_ctx"]["title"])

        pos_num = len(samples)

        total_txt = positive_txt + hard_neg_txt
        total_title = positive_title + hard_neg_title

        question_encode = self.tokenizer(
            question, padding=self.padding, truncation=self.truncation, max_length=self.max_length
        )

        context_encode = self.tokenizer(
            total_title, total_txt, padding=self.padding, truncation=self.truncation, max_length=self.max_length
        )

        # question_batch: (batch size, sequence_length)
        question_batch = {
            "input_ids": torch.tensor(question_encode["input_ids"]),
            "attention_mask": torch.tensor(question_encode["attention_mask"]),
            "token_type_ids": torch.tensor(question_encode["token_type_ids"]),
        }

        features = {}
        for key in context_encode:
            features[key] = [
                (
                    [context_encode[key][i], context_encode[key][i + pos_num]]
                    if len(hard_neg_txt) > 0
                    else [context_encode[key][i]]
                )
                for i in range(pos_num)
            ]

        # context_batch : (batch size, 2 or 1, sequence_length)
        context_batch = {
            "input_ids": torch.tensor(features["input_ids"]),
            "attention_mask": torch.tensor(features["attention_mask"]),
            "token_type_ids": torch.tensor(features["token_type_ids"]),
        }

        return question_batch, context_batch
