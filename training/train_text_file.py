import pandas as pd
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
)
import re
import os

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.bos_token = "<|startoftext|>"
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token


# Load and preprocess data
os.chdir("/Users/scandukuri/CS230-SPRING23/")
csv_file = "data/processed/train.csv"
chunksize = 1000  # Define the desired chunk size

pattern = r"\[.*?\]"

# Open the output file
with open("data/processed/train_lyrics_data.txt", "w", encoding="utf-8") as f:
    # Iterate over the DataFrame in chunks
    for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunksize)):
        print(f"Processing rows {i * 1000} to {i * 1000 + 999}")
        chunk["formatted_text"] = (
            f"{tokenizer.bos_token} Genre: "
            + chunk["tag"]
            + " Title: "
            + chunk["title"]
            + " Lyrics: "
            + chunk["lyrics"]
            + f" {tokenizer.eos_token}"
        )

        text_data = chunk["formatted_text"].tolist()

        # Remove tags like [Chorus: ] etc.
        for line in text_data:
            if isinstance(line, str):
                line = re.sub(r"\[.*?\]", "", line)
                line = re.sub(r"\(.*?chorus.*?\)", "", line, flags=re.IGNORECASE)
                line = re.sub(r"\(.*?verse.*?\)", "", line, flags=re.IGNORECASE)
                line = re.sub(r"\(.*?hook.*?\)", "", line, flags=re.IGNORECASE)
                line = re.sub(r"\(.*?intro.*?\)", "", line, flags=re.IGNORECASE)
                line = re.sub(r"\(.*?outro.*?\)", "", line, flags=re.IGNORECASE)
                line = re.sub(r"\(.*?pre.*?\)", "", line, flags=re.IGNORECASE)
                line = re.sub(r"\(.*?bridge.*?\)", "", line, flags=re.IGNORECASE)
                line = re.sub(r"\[.*?\]", " ", line)
                line = re.sub(r"[^\w\s:]+", " ", line)
                f.write(f"{line}\n")
