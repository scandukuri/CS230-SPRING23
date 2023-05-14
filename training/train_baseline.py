import pandas as pd
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
)
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW
import re
import os


device = torch.device("cuda")
model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2").to(device)
tokenizer.bos_token = "<|startoftext|>"
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


# Load and preprocess data
# os.chdir("/Users/scandukuri/CS230-SPRING23/")
csv_file = "data/processed/train.csv"
data = pd.read_csv(csv_file)
data["formatted_text"] = (
    f"{tokenizer.bos_token} Genre: "
    + data["tag"]
    + " Title: "
    + data["title"]
    + " Lyrics: "
    + data["lyrics"]
    + f" {tokenizer.eos_token}"
)


pattern = r"\[.*?\]"
text_data = data["formatted_text"].tolist()

# remove tags like [Chorus: ] etc


def save_text_data(text_data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for line in text_data:
            if isinstance(line, str):
                line = re.sub(r"\[.*?\]", "", line)
                line = re.sub(r"\(.*?Chorus.*?\)", "", line)
                line = re.sub(r"\(.*?Verse.*?\)", "", line)
                line = re.sub(r"\(.*?Hook.*?\)", "", line)
                line = re.sub(r"\(.*?Intro.*?\)", "", line)
                line = re.sub(r"\(.*?Outro.*?\)", "", line)
                line = re.sub(r"\(.*?Pre-.*?\)", "", line)
                line = re.sub(r"\(.*?chorus.*?\)", "", line)
                line = re.sub(r"\(.*?verse.*?\)", "", line)
                line = re.sub(r"\(.*?hook.*?\)", "", line)
                line = re.sub(r"\(.*?intro.*?\)", "", line)
                line = re.sub(r"\(.*?outro.*?\)", "", line)
                line = re.sub(r"\(.*?pre-.*?\)", "", line)
                f.write(f"{line}\n")


save_text_data(text_data, "data/processed/train_lyrics_data.txt")


train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/processed/train_lyrics_data.txt",
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


training_args = TrainingArguments(
    output_dir="models/baseline/train",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        labels_shifted = labels[..., 1:].contiguous()
        logits_shifted = logits[..., :-1, :].contiguous()
        loss = loss_fct(
            logits_shifted.view(-1, logits_shifted.size(-1)), labels_shifted.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model("models/baseline/train")
tokenizer.save_pretrained("models/baseline/train")
