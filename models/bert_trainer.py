import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import tqdm
from superprompt.models.bert import BERT
import wandb
from transformers import BertTokenizer, DataCollatorForLanguageModeling
import random
from datasets import IterableDataset
from utils import get_available_device, sample_prompt_pairs


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min(
            [
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps,
            ]
        )

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class BERTTrainer:
    def __init__(
        self,
        bert: BERT,
        tokenizer: BertTokenizer,
        collator: DataCollatorForLanguageModeling,
        train_dataset: IterableDataset,
        test_dataset: IterableDataset,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        max_len: int = 256,
        batch_size: int = 32,
        log_freq: int = 10,
        valid_freq: int = 10,
        save_freq: int = 1000,
        output_path: str = "./saved",
        use_wandb: bool = False,
    ):
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        # This BERT model will be saved every epoch
        # Initialize the BERT Language Model, with BERT model
        self.device = get_available_device()
        self.model = bert.to(self.device)

        self.tokenizer = tokenizer
        self.collator: DataCollatorForLanguageModeling = collator
        self.use_wandb: bool = use_wandb
        self.batch_size: int = batch_size
        self.output_path: str = output_path

        # Setting the train and test data loader
        self.train_data = train_dataset
        self.test_data = test_dataset
        self.max_len = max_len

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, 10)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        print("mask token id", tokenizer.mask_token_id)
        self.criterion = nn.NLLLoss(ignore_index=tokenizer.mask_token_id)

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.valid_freq = valid_freq
        self.table_rows = []

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, dataset: IterableDataset, train=True):
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(dataset.iter(batch_size=self.batch_size)),
            bar_format="{l_bar}{r_bar}",
        )

        avg_loss = 0.0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            collated = self.collator(data["input_ids"])
            input_ids = collated["input_ids"].to(self.device)
            attn_mask = torch.stack(data["attention_mask"]).to(self.device)

            mask_lm_output = self.model.forward(input_ids, attn_mask)

            transposed_output = mask_lm_output.transpose(1, 2)

            loss = self.criterion(transposed_output, input_ids)

            avg_loss += loss.item()
            avg_loss /= i + 1
            print(f"epoch {epoch} i {i} avg_loss {avg_loss}")

            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if i % self.log_freq == 0:
                post_fix = {
                    "epoch": epoch,
                    "batch": i,
                    "avg_loss": avg_loss,
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))
                if self.use_wandb:
                    wandb.log(post_fix)
            if i % self.valid_freq == 0:
                decoded = self.eval_sample()
                if self.use_wandb:
                    self.table_rows.append([epoch, avg_loss, decoded])
                    print("table", len(self.table_rows))
                    table = wandb.Table(
                        data=self.table_rows,
                        columns=["epoch", "avg_loss", "sample"],
                    )
                    wandb.log({"samples": table})
            if i % self.save_freq == 0:
                self.save(epoch, self.output_path)

    def eval_sample(self):
        prompt = random.choice(sample_prompt_pairs)
        print("---EVAL---")
        print("prompt", prompt)
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        print("tokenized", tokenized)
        eval_batch = self.collator([tokenized])
        print("batch", eval_batch)
        input_ids = eval_batch["input_ids"].squeeze(0).to(self.device)
        attn_mask = tokenized["attention_mask"].to(self.device)
        print("input ids", input_ids)
        mask_lm_output = self.model.forward(input_ids, attn_mask)
        output = torch.argmax(mask_lm_output, dim=2)
        print("output", output)
        decoded = self.tokenizer.decode(output[0])
        print("decoded", decoded)
        return decoded

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
