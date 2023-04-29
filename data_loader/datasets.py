import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, GPT2Tokenizer

import re


class LyricsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = self.data.iloc[idx]["title"]
        tag = self.data.iloc[idx]["tag"]
        lyrics = self.data.iloc[idx]["lyrics"]

        # Perform preprocessing on the lyrics
        lyrics = re.sub(
            r"\[Produced .*?\]", "", lyrics
        )  # remove tags like [Produced by]

        lyrics = re.sub(
            r":.*?\]", ":]", lyrics
        )  # remove any possible artist names after "Chorus: " or "Verse: " etc.

        lyrics = re.sub(
            r"\n{3,}", "\n\n", lyrics
        )  # replace multiple newlines with two newlines

        lyrics = re.sub(
            r'^\n+', '', lyrics
        )

        # Tokenize the preprocessed lyrics
        inputs = self.tokenizer.encode_plus(
            lyrics.strip(),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Return the tokenized lyrics and any additional metadata
        return {
            "title": title,
            "tag": tag,
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
        }
