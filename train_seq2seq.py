from argparse import Namespace
import torch
import torch.nn as nn
from seq2seq_model import Encoder, Decoder, Seq2Seq
from utils import get_available_device
from torch.optim import AdamW
import time
import math
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    DataCollatorForSeq2Seq,
)
import torch.utils.data as data


class Args(Namespace):
    input_dim = 128
    output_dim = 128
    enc_emb_dim = 256
    dec_emb_dim = 256
    hid_dim = 512
    n_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5
    n_epochs = 10
    clip = 1
    max_length = 128
    batch_size = 128


dataset = load_dataset(
    "roborovski/diffusiondb-masked-no-descriptors",
    streaming=True,
)
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", use_fast=True
)
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
dataset = dataset.map(
    lambda x: tokenizer(
        x["prompt"],
        truncation=True,
        padding="max_length",
        max_length=Args.max_length,
        return_tensors="pt",
    ),
    batched=True,
)

enc = Encoder(
    Args.input_dim, Args.enc_emb_dim, Args.hid_dim, Args.n_layers, Args.enc_dropout
)
dec = Decoder(
    Args.output_dim, Args.dec_emb_dim, Args.hid_dim, Args.n_layers, Args.dec_dropout
)

device = get_available_device()
model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)

optimizer = AdamW(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.mask_token_id)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = tokenizer(
            batch["masked"],
            truncation=True,
            padding="max_length",
            max_length=Args.max_length,
            return_tensors="pt",
        )['input_ids']
        trg = tokenizer(
            batch["prompt"],
            truncation=True,
            padding="max_length",
            max_length=Args.max_length,
            return_tensors="pt",
        )['input_ids']
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float("inf")


for epoch in range(Args.n_epochs):
    start_time = time.time()

    train_loss = train(model, dataset["train"], optimizer, criterion, Args.clip)
    valid_loss = evaluate(model, dataset["test"], criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")

    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")
