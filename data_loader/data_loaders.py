import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, GPT2Tokenizer
from datasets import LyricsDataset


class LyricsDataLoader(DataLoader):
    def __init__(
        self, data_path, tokenizer, max_length, batch_size, shuffle=True, num_workers=1
    ):
        dataset = LyricsDataset(data_path, tokenizer, max_length)
        super().__init__(dataset, batch_size, shuffle, num_workers)
