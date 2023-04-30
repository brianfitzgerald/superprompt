from argparse import Namespace
from torch.utils.data import DataLoader
import gc
import torch
from bert.model import BERT
from bert.trainer import BERTTrainer
import wandb
from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorForLanguageModeling
import sys

gc.collect()

args = Namespace(
    hidden=256,
    batch_size=32,
    layers=8,
    attn_heads=8,
    adam_weight_decay=0.01,
    adam_beta1=0.9,
    output_path="/home/ubuntu/superprompt/saved",
    epochs=50,
    log_freq=50,
    save_freq=32 * 10,
    valid_freq=32 * 100,
    adam_beta2=0.999,
    num_workers=4,
    lr=1e-3,
    max_len=256,
    use_wandb=True,
)

if __name__ != "__main__":
    sys.exit(0)


dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", streaming=True,)
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", use_fast=True
)
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.25
)
dataset = dataset.map(
    lambda x: tokenizer(
        x["Prompt"],
        truncation=True,
        padding="max_length",
        max_length=args.max_len,
        return_tensors="pt",
    ),
    batched=True,
)
dataset = dataset.remove_columns(["Prompt"])

print(args)


print("Building BERT model")

first_batch = next(iter(dataset["train"]))
collated = collator([first_batch])

label_list = list(collated["labels"].numpy())


bert = BERT(
    tokenizer.vocab_size,
    hidden=args.hidden,
    n_layers=args.layers,
    attn_heads=args.attn_heads,
    max_len=args.max_len,
)

if args.use_wandb:
    wandb.init(config=args, project="superprompt")
    wandb.watch(bert, log_freq=args.log_freq)

print("Creating BERT Trainer")
trainer = BERTTrainer(
    bert,
    tokenizer,
    collator,
    dataset["train"],
    dataset["test"],
    args.lr,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    max_len=args.max_len,
    log_freq=args.log_freq,
    valid_freq=args.valid_freq,
    batch_size=args.batch_size,
    use_wandb=args.use_wandb,
)

for epoch in range(args.epochs):
    trainer.train(epoch)
    if epoch % args.save_freq == 0:
        trainer.save(epoch, args.output_path)

    if dataset["test"] is not None and epoch % args.valid_freq == 0:
        trainer.test(epoch)
