import pandas as pd
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
)
from transformers import Trainer, TrainingArguments, logging
from torch.optim import AdamW
import re
import os
import datasets

dataset = datasets.load_dataset("text", data_dir="trainingdataset")["train"]
max_seq_length = 128
num_proc = 8

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")


device = torch.device("cuda")
model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.bos_token = "<|startoftext|>"
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    # Remove empty lines
    examples["text"] = [
        line for line in examples["text"] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=["text"],
)


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop,
    # you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result


train_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=num_proc,
)

print("Loaded training file into text dataset successfully.")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="models",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=500000,
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(inputs["input_ids"], labels=labels)
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
trainer.save_model("models")
tokenizer.save_pretrained("models")
