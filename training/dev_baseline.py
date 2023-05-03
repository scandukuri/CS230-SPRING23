import pandas as pd
from datasets import Dataset
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


model = GPT2LMHeadModel.from_pretrained("distilgpt2")
device = torch.device("cuda")
model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.bos_token = "<|startoftext|>"
tokenizer.eos_token = "<|startoftext|>"
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


# Load and preprocess data
# os.chdir("/Users/scandukuri/CS230-SPRING23/")
csv_file = "data/processed/dev.csv"
data = pd.read_csv(csv_file)
data["formatted_text"] = (
    f"{tokenizer.bos_token}"
    + " Genre: "
    + data["tag"]
    + " Title: "
    + data["title"]
    + " Lyrics: "
    + data["lyrics"]
    + f" {tokenizer.eos_token}"
)

pattern = r"\[.*?\]"
data["formatted_text"] = data["formatted_text"].apply(lambda x: re.sub(pattern, "", x))
text_data = data["formatted_text"].tolist()
# remove tags like [Chorus: ] etc


def save_text_data(text_data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for line in text_data:
            f.write(f"{line}\n")


save_text_data(text_data, "data/processed/dev_lyrics_data.txt")


train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/processed/dev_lyrics_data.txt",
    block_size=128,
)

train_dataset = train_dataset.map(
    lambda x: {
        "input_ids": x["input_ids"].to(device),
        "attention_mask": x["attention_mask"].to(device),
    }
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


training_args = TrainingArguments(
    output_dir="models/baseline/dev",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=100_000,
    device=device,
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
trainer.save_model("models/baseline/dev")
tokenizer.save_pretrained("models/baseline/dev")
