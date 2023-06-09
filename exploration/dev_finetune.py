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
import os

os.chdir("/Users/scandukuri/CS230-SPRING23/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

# Load and preprocess data
csv_file = "data/processed/dev.csv"
data = pd.read_csv(csv_file)
data["concat"] = data["tag"] + " " + data["title"] + " " + data["lyrics"]
text_data = data["concat"].tolist()


def save_text_data(text_data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for line in text_data:
            f.write(f"{line}\n")


save_text_data(text_data, "lyrics_data.txt")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


tokenizer.pad_token = tokenizer.eos_token


train_dataset = TextDataset(
    tokenizer=tokenizer, file_path="lyrics_data.txt", block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./output_dir",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=500,
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

trainer.save_model("./output_dir")
tokenizer.save_pretrained("./output_dir")
