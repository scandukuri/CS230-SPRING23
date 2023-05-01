import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv

import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LyricDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lyrics = []

        for idx in range(len(self.data)):
            title = self.data.iloc[idx]["title"]
            genre = self.data.iloc[idx]["tag"]
            lyrics = self.data.iloc[idx]["lyrics"]

            lyrics = re.sub(
            r"\[.*?\]", "", lyrics
            )  # remove tags like [Produced by]

            lyrics = re.sub(
                r'\n+', '\n', lyrics
            ).strip()

            lyrics = re.sub(
                r'!\n', '! ', lyrics
            ).strip()

            lyrics = re.sub(
                r'\?\n', '? ', lyrics
            ).strip()

            lyrics = re.sub(
                r',\n', ', ', lyrics
            ).strip()

            lyrics = re.sub(
                r';\n', '; ', lyrics
            ).strip()

            lyrics = re.sub(
                r':\n', ': ', lyrics
            ).strip()

            lyrics = re.sub(
                r'\n', '. ', lyrics
            ).strip()
            
            prompt = f'Genre {genre} Title {title} Lyrics '
            prompt_tokenized = tokenizer.encode(prompt, padding='do_not_pad', add_special_tokens=False, return_tensors='pt').squeeze(0)
            lyrics_tokenized = tokenizer.encode(lyrics, padding='max_length', max_length=max_length - prompt_tokenized.shape[0], return_tensors='pt', add_special_tokens=False).squeeze(0)
            tokenized = torch.cat((prompt_tokenized, lyrics_tokenized))
            attention_mask = (tokenized != tokenizer.eos_token_id).int()
            labels = tokenized.clone()
            labels[:prompt_tokenized.shape[0]] = -100
            labels[labels == tokenizer.eos_token_id] = -100
            
            self.lyrics.append({ 'input_ids' : tokenized, 'attention_mask' : attention_mask, 'labels' : labels})


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.lyrics[idx]


class LyricsDataLoader(DataLoader):
    def __init__(
        self, data_path, tokenizer, max_length, batch_size, num_workers=1
    ):
        dataset = LyricsDataset(data_path, tokenizer, max_length)
        super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)