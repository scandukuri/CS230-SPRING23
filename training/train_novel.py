import pandas as pd
import torch
from torch import nn
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
)
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW
import re
import nltk
from nltk import ngrams
from nltk.lm import MLE
import math
from torch.nn import functional as F
import torch.nn.init as init
import gc
import pickle


# Load pre-trained model (weights)
baseline_model = "models/finalmodel"
device = torch.device("cuda")
model = GPT2LMHeadModel.from_pretrained(baseline_model).to(device)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("Model and tokenizer loaded.")

# Freeze GPT-2 weights
for param in model.parameters():
    param.requires_grad = False


class CustomModel(nn.Module):
    def __init__(self, gpt2_model):
        super(CustomModel, self).__init__()
        self.gpt2 = gpt2_model
        self.linear1 = nn.Linear(50257, 768)
        self.linear2 = nn.Linear(768, 768)
        self.linear3 = nn.Linear(768, 768)
        self.linear4 = nn.Linear(768, 768)
        self.linear5 = nn.Linear(768, 768)
        self.linear6 = nn.Linear(768, 50257)

    def forward(self, inputs):
        x = self.gpt2(inputs)[0]  # Get the output logits tensor
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        x = nn.ReLU()(self.linear3(x))
        x = nn.ReLU()(self.linear4(x))
        x = nn.ReLU()(self.linear5(x))
        x = self.linear6(x)
        return x


# Instantiate your custom model with GPT-2 as the base model
custom_model = CustomModel(model).to(device)

print("Model instantiated.")
print(custom_model)


train = pd.read_csv("train_smaller.csv")[["title", "tag", "lyrics"]][:500000]
train["prompt"] = "Genre: " + train["tag"] + " Title: " + train["title"] + " Lyrics: "
train.drop(columns=["title", "tag"], inplace=True)
# Define the desired order of columns
new_order = ["prompt", "lyrics"]

# Reorder the columns using reindex()
train = train.reindex(columns=new_order)


def calculate_ttr(text):
    tokens = text.split()
    token_count = len(tokens)
    unique_tokens = len(set(tokens))
    ttr = unique_tokens / token_count
    return ttr


def calculate_perplexity(text, n=2):
    tokens = text.split()
    ngrams_list = list(ngrams(tokens, n))
    model = MLE(n)
    model.fit([ngrams_list], vocabulary_text=tokens)
    perplexity_score = model.perplexity(ngrams_list)
    return perplexity_score


def insert_newlines(text):
    # Find all matches of two words without space between
    matches = re.findall(r"([a-z])([A-Z])", text)

    # For each match, insert a newline between the two words
    for match in matches:
        text = text.replace("".join(match), match[0] + "\n" + match[1])

    return text


def compute_reward(generated_sequence, target_sequence):
    try:
        reward = -(
            (
                (
                    calculate_ttr(generated_sequence)
                    - calculate_perplexity(generated_sequence)
                )
                - (
                    calculate_ttr(target_sequence)
                    - calculate_perplexity(target_sequence)
                )
            )
            ** 2
        )
        perplexity = calculate_perplexity(generated_sequence)
        ttr = calculate_ttr(generated_sequence)
    except:
        return None
    return reward, perplexity, ttr


optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.00000005)
metrics = {"index": [], "loss": [], "ttr": [], "perplexity": []}

checkpoint_interval = 10000
print("Beginning training.")
for index, row in train.iterrows():
    if index % checkpoint_interval == 0 and index > 0:
        # Save model checkpoint
        torch.save(custom_model.state_dict(), f"novelmodel_checkpoint_{index}.pth")

        # Pickle out metrics dictionary

        with open(f"novelmodel_metrics_checkpoint_{index}.pkl", "wb") as f:
            pickle.dump(metrics, f)
    print(index)
    prompt, target_sequence = tuple(row)
    line = target_sequence
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
    line = re.sub(r"\bx\s*\d+\b", "", line, flags=re.IGNORECASE)
    target_sequence = line

    # Generate a sequence of tokens and log the probabilities
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    log_probs_sum = 0.0

    for _ in range(100):  # Generate 10 tokens
        # Get logits for the next token
        outputs = custom_model(input_ids)
        next_token_logits = outputs[:, -1, :]

        # Apply softmax to logits
        probs = F.softmax(next_token_logits, dim=-1)

        # Sample a token
        next_token = torch.multinomial(probs, num_samples=1)
        while next_token == tokenizer.bos_token or next_token == tokenizer.eos_token:
            next_token = torch.multinomial(probs, num_samples=1)
        # Append the token to the input sequence
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        log_probs_next_token = torch.log(probs)

        # Add the log probability of the selected token
        log_probs_sum += torch.log(probs[0, next_token.item()])

    # Decode the input sequence to text
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)[
        len(prompt) :
    ]

    processed_text = " ".join([word for word in output_text.split()])
    processed_text = insert_newlines(processed_text)
    # Compute the reward
    result = compute_reward(processed_text, target_sequence)
    if not result:
        continue
    reward, perplexity, ttr = result[0], result[1], result[2]

    # Compute the loss: -1 * reward * sum(log_probabilities)
    loss = -reward * log_probs_sum
    metrics["index"].append(index)
    metrics["loss"].append(float(loss))
    metrics["perplexity"].append(perplexity)
    metrics["ttr"].append(ttr)
    # Backpropagate the loss
    loss.backward()

    # Update the weights
    optimizer.step()

    # Zero the gradients
    optimizer.zero_grad()

    # Delete intermediate tensors to free up memory
    del input_ids, log_probs_sum, reward, loss
    gc.collect()

torch.save(custom_model, "novelmodel.pth")
df = pd.DataFrame(metrics)
df.to_pickle("metrics.pkl")
