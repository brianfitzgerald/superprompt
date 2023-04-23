from argparse import Namespace
from torch.utils.data import DataLoader
import gc
import torch
from bert.model import BERT
from bert.trainer import BERTTrainer
import wandb

gc.collect()

torch.cuda.empty_cache()


args = Namespace(
    hidden=256,
    batch_size=64,
    layers=8,
    attn_heads=8,
    adam_weight_decay=0.01,
    adam_beta1=0.9,
    output_path="/home/ubuntu/superprompt/saved",
    epochs=50,
    log_freq=50,
    adam_beta2=0.999,
    cuda_devices=[0],
    num_workers=4,
    lr=1e-3,
    with_cuda=True,
    valid_freq=5,
)

from datasets import load_dataset
from transformers import BertTokenizer

dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts")
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", use_fast=True
)
dataset = dataset.map(
    lambda x: tokenizer(
        x["Prompt"], truncation=True, padding="max_length", max_length=512
    ),
    batched=True,
)
dataset = dataset.with_format("torch")

print(args)


print("Building BERT model")

bert = BERT(
    tokenizer.vocab_size,
    hidden=args.hidden,
    n_layers=args.layers,
    attn_heads=args.attn_heads,
)


wandb.init(config=args, project="superprompt")
wandb.watch(bert, log_freq=args.log_freq)


train_dataloader = DataLoader(
    dataset["train"], batch_size=args.batch_size, num_workers=args.num_workers
)
test_dataloader = DataLoader(
    dataset["test"], batch_size=args.batch_size, num_workers=args.num_workers
)

print("Creating BERT Trainer")
trainer = BERTTrainer(
    bert,
    tokenizer.vocab_size,
    tokenizer=tokenizer,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    lr=args.lr,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    log_freq=args.log_freq,
    with_cuda=args.with_cuda,
    cuda_devices=args.cuda_devices,
)

for epoch in range(args.epochs):
    trainer.train(epoch)
    trainer.save(epoch, args.output_path)

    if test_dataloader is not None:
        trainer.test(epoch)
