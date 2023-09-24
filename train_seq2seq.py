from argparse import Namespace
from enum import IntEnum
import torch
import torch.nn as nn
from utils import (
    get_available_device,
    should_use_wandb,
    sample_prompt_pairs,
    sample_translate_pairs,
)
from seq2seq_model import Attention, Encoder, Decoder, Seq2Seq
from torch.optim import AdamW
import time
import math
from datasets import load_dataset, Dataset, ReadInstruction
from transformers import (
    BertTokenizer,
    DataCollatorForSeq2Seq,
)
import random
import wandb
import numpy as np
from torch.utils.data import DataLoader

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Task(IntEnum):
    DIFFUSION = 1
    TRANSLATE = 2


class Args(Namespace):
    enc_emb_dim = 256
    dec_emb_dim = 256
    enc_hid_dim = 512
    dec_hid_dim = 512
    enc_dropout = 0.5
    dec_dropout = 0.5
    n_epochs = 10
    clip = 1
    max_length = 64
    batch_size = 64
    use_wandb = should_use_wandb()
    log_freq = 2
    # this is in samples
    valid_freq = 128
    task = Task.DIFFUSION.value
    sample_limit = 10e5

max_length = Args.max_length

def tokenize_batch(batch):
    src = [f"[BOS] {s} [EOS]" for s in batch["src"]]
    src = tokenizer(
        batch["src"],
        truncation=True,
        return_length=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    trg = [f"[BOS] {s} [EOS]" for s in batch["trg"]]
    trg = tokenizer(
        batch["trg"],
        truncation=True,
        return_length=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "src_input_ids": src["input_ids"],
        "src_len": src["length"],
        "trg_input_ids": trg["input_ids"],
    }


tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", use_fast=True
)
tokenizer.add_special_tokens(
    {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
)

print("Task: ", Args.task)
valid_src = []

if Args.task == Task.DIFFUSION.value:
    dataset = load_dataset(
        "roborovski/diffusiondb-masked-no-descriptors",
        split=ReadInstruction("train", to=10, unit="%")
    )
    dataset = dataset.rename_columns({"masked": "src", "prompt": "trg"})
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=256,
        remove_columns=["src", "trg"],
    )
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=0.1)
    valid_dataset = Dataset.from_dict(
        {
            "src": [x[0] for x in sample_prompt_pairs],
            "trg": [x[1] for x in sample_prompt_pairs],
        }
    )
elif Args.task == Task.TRANSLATE.value:
    dataset = load_dataset("bentrevett/multi30k")
    dataset = dataset.rename_columns({"de": "src", "en": "trg"})
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=Args.batch_size,
        remove_columns=["src", "trg"],
    )
    valid_dataset = Dataset.from_dict(
        {
            "src": [x[0] for x in sample_translate_pairs],
            "trg": [x[1] for x in sample_translate_pairs],
        }
    )

valid_src = [valid_dataset["src"], valid_dataset["trg"]]
valid_dataset = tokenize_batch(valid_dataset)

input_dim_size = tokenizer.vocab_size
attn = Attention(Args.enc_hid_dim, Args.dec_hid_dim)
enc = Encoder(
    input_dim_size,
    Args.enc_emb_dim,
    Args.enc_hid_dim,
    Args.dec_hid_dim,
    Args.enc_dropout,
)
dec = Decoder(
    input_dim_size,
    Args.dec_emb_dim,
    Args.enc_hid_dim,
    Args.dec_hid_dim,
    Args.dec_dropout,
    attn,
)

device = get_available_device()
model = Seq2Seq(enc, dec, tokenizer.pad_token_id, device).to(device)

print("Device: ", device)


def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = AdamW(model.parameters(), lr=3e-4)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


def train(model: Seq2Seq, dataset: Dataset, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    loader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=True)

    for i, batch in enumerate(loader):
        if i > Args.sample_limit:
            print("Sample limit reached, returning.")
            break
        src_input_ids = torch.stack(batch["src_input_ids"]).to(device)
        trg_input_ids = torch.stack(batch["trg_input_ids"]).to(device)
        src_len = batch["src_len"].to(device)

        optimizer.zero_grad()

        output = model(src_input_ids, src_len, trg_input_ids)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg_input_ids = trg_input_ids[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg_input_ids)
        # loss.requires_grad = True
        print(f"iteration {i}: {loss.item()}")
        if i % Args.log_freq == 0 and Args.use_wandb:
            wandb.log({"loss": loss.item()})

        if i % Args.valid_freq == 0:
            validate(model, i, epoch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataset)


def evaluate(model: Seq2Seq, dataset: Dataset, criterion):
    model.eval()

    epoch_loss = 0
    loader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=True)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            src_input_ids = torch.stack(batch["src_input_ids"]).to(device)
            trg_input_ids = torch.stack(batch["trg_input_ids"]).to(device)
            src_len = batch["src_len"].to(device)

            optimizer.zero_grad()

            output = model(src_input_ids, src_len, trg_input_ids)

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg_input_ids = trg_input_ids[1:].view(-1)
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            if output.shape[0] != trg_input_ids.shape[0]:
                print(
                    "output shape : ",
                    output.shape,
                    " does not match trg_input_ids: ",
                    trg_input_ids.shape,
                )
                continue
            loss = criterion(output, trg_input_ids)

            epoch_loss += loss.item()

    return epoch_loss / len(dataset)


def validate(model: Seq2Seq, idx: int, epoch: int):
    model.eval()

    with torch.no_grad():
        src_input_ids = valid_dataset["src_input_ids"].transpose(1, 0).to(device)
        trg_input_ids = valid_dataset["trg_input_ids"].transpose(1, 0).to(device)
        src_len = valid_dataset["src_len"].to(device)

        outputs = model(src_input_ids, src_len, trg_input_ids)

        # Create a target tensor with batch size 1 and length max_length with all tokens masked
        outputs = outputs.argmax(dim=-1)
        outputs = outputs.transpose(1, 0)
        output_ls = outputs.squeeze().tolist()
        outputs = [tokenizer.decode(x) for x in output_ls]
        for i in range(len(valid_src)):
            print("Input: ", valid_src[0][i], "Expected: ", valid_src[1][i], "Output: ", outputs[i])
            valid_table_data.append(
                [
                    epoch,
                    idx,
                    valid_src[0][i],
                    valid_src[1][i],
                    outputs[i],
                ]
            )
    if Args.use_wandb:
        sample_table = wandb.Table(
            columns=["epoch", "idx", "input", "expected", "output"],
            data=valid_table_data,
        )
        wandb.log({"sample": sample_table})
    model.train()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float("inf")

if Args.use_wandb:
    wandb.init(config=Args, project="superprompt-seq2seq-rnn")
    wandb.watch(model, log_freq=Args.log_freq)
    print("wandb initialized")

valid_table_data = []

for epoch in range(Args.n_epochs):
    start_time = time.time()

    train_loss = train(model, dataset["train"], optimizer, criterion, Args.clip)
    print("train_loss: ", train_loss)
    eval_loss = evaluate(model, dataset["test"], criterion)
    if Args.use_wandb:
        wandb.log(
            {
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "PPL": math.exp(train_loss),
            }
        )
    print("eval_loss", eval_loss)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if eval_loss < best_valid_loss:
        best_valid_loss = eval_loss
        task = Args.task
        torch.save(model.state_dict(), f"model-{epoch}-task{task}.pt")

    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
    print(f"\t Val. Loss: {eval_loss:.3f} |  Val. PPL: {math.exp(eval_loss):7.3f}")
