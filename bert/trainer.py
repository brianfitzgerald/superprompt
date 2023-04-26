import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from IPython.display import display
from bert.model import BERT
import wandb
from transformers import BertTokenizer
import random

def mask_random_word(batch, tokenizer: BertTokenizer):

    for i, p in enumerate(batch["Prompt"]):

        prompt = p.split(" ")

        for i, token in enumerate(prompt):
            prob = random.random()
            if prob < 0.15:
                prompt[i] = tokenizer.mask_token

        batch["Prompt"][i] = " ".join(prompt)

    return batch


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
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        warmup_steps=10000,
        with_cuda: bool = True,
        cuda_devices=None,
        log_freq: int = 10,
    ):
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        # Initialize the BERT Language Model, with BERT model
        self.model = bert.to(self.device)

        self.tokenizer = tokenizer

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.hidden, n_warmup_steps=warmup_steps
        )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        self.text_table = wandb.Table(columns=["epoch", "loss", "text"])


    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (str_code, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        avg_loss = 0.0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            input_ids = data["input_ids"].to(self.device)

            # 1. forward the next_sentence_prediction and masked_lm model
            mask_lm_output = self.model.forward(input_ids)

            # 2-2. NLLLoss of predicting masked token word
            loss = self.criterion(mask_lm_output.transpose(1, 2), input_ids)

            # next sentence prediction accuracy
            avg_loss += loss.item()

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            if i % self.log_freq == 0:
                output = torch.argmax(mask_lm_output, dim=2)
                decoded = self.tokenizer.decode(output[0])
                self.text_table.add_data(epoch, loss.item(), decoded)
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                    "sample": self.text_table
                }
                data_iter.write(str(post_fix))
                wandb.log(post_fix)
                print(
                    "EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter)
                )

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
