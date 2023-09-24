from argparse import Namespace
import gc
from bert.model import BERT
from bert.trainer import BERTTrainer
import wandb
from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorForLanguageModeling
import sys
import os

from utils import should_use_wandb, sample_prompt_pairs


gc.collect()


class Args(Namespace):
    hidden = 256
    batch_size = 64
    layers = 8
    attn_heads = 8
    adam_weight_decay = 0.01
    adam_beta1 = 0.9
    output_path = "/home/ubuntu/superprompt/saved"
    epochs = 500
    log_freq = 32 * 2
    save_freq = 32 * 10
    valid_freq = 32 * 5
    adam_beta2 = 0.999
    num_workers = 4
    lr = 3e-4
    max_len = 128
    use_wandb = should_use_wandb()


if __name__ != "__main__":
    sys.exit(0)

dataset = load_dataset(
    "Gustavosta/Stable-Diffusion-Prompts",
    streaming=True,
)
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", use_fast=True
)
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
dataset = dataset.map(
    lambda x: tokenizer(
        x["Prompt"],
        truncation=True,
        padding="max_length",
        max_length=Args.max_len,
        return_tensors="pt",
    ),
    batched=True,
)
dataset = dataset.remove_columns(["Prompt"])


bert = BERT(
    tokenizer.vocab_size,
    hidden=Args.hidden,
    n_layers=Args.layers,
    attn_heads=Args.attn_heads,
    max_len=Args.max_len,
)

if Args.use_wandb:
    wandb.init(config=Args, project="superprompt")
    wandb.watch(bert, log_freq=Args.log_freq)

print("Creating BERT Trainer")
trainer = BERTTrainer(
    bert,
    tokenizer,
    collator,
    dataset["train"],
    dataset["test"],
    Args.lr,
    betas=(Args.adam_beta1, Args.adam_beta2),
    weight_decay=Args.adam_weight_decay,
    max_len=Args.max_len,
    log_freq=Args.log_freq,
    valid_freq=Args.valid_freq,
    batch_size=Args.batch_size,
    use_wandb=Args.use_wandb,
)

for epoch in range(Args.epochs):
    trainer.train(epoch)
