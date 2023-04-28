import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from IPython.display import display
from bert.model import BERT
import wandb
from transformers import BertTokenizer, DataCollatorForLanguageModeling
import random
from datasets import IterableDataset
sample_prompts = [
    "human sculpture of lanky tall alien on a romantic date at italian restaurant with smiling woman, nice restaurant, photography, bokeh",
    "portrait of barbaric spanish conquistador, symmetrical, by yoichi hatakenaka, studio ghibli and dan mumford",
    "a small liquid sculpture, corvette, viscous, reflective, digital art",
    "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render.",
    "cyber moai on easter island, digital painting, highly detailed, concept art, trending on artstation, epic composition, sharp focus, flawless render, masterpiece, volumetric lighting"
]

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
        warmup_steps=10000,
        max_len: int = 256,
        log_freq: int = 10,
        use_wandb: bool = False,
    ):
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        self.device = device
        print('device', device)
        
        # This BERT model will be saved every epoch
        # Initialize the BERT Language Model, with BERT model
        self.model = bert.to(self.device)

        self.tokenizer = tokenizer
        self.collator = collator
        self.use_wandb = use_wandb

        # Setting the train and test data loader
        self.train_data = train_dataset
        self.test_data = test_dataset
        self.max_len = max_len

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.hidden, n_warmup_steps=warmup_steps
        )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        print('mask token id', tokenizer.mask_token_id)
        self.criterion = nn.NLLLoss(ignore_index=tokenizer.mask_token_id)

        self.log_freq = log_freq
        self.table_rows = []


    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, dataset: IterableDataset, train=True):
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(dataset),
            desc="EP_%s:%d" % (str_code, epoch),
            bar_format="{l_bar}{r_bar}",
        )

        avg_loss = 0.0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            collated = self.collator([data])
            input_ids = collated["input_ids"].to(self.device)
            labels = collated["labels"].to(self.device)

            # 1. forward the next_sentence_prediction and masked_lm model
            mask_lm_output = self.model.forward(input_ids)

            transposed_output = mask_lm_output.transpose(1, 2)
            output = torch.argmax(transposed_output, dim=2)

            # 2-2. NLLLoss of predicting masked token word
            loss = self.criterion(transposed_output, input_ids)

            # next sentence prediction accuracy
            avg_loss += loss.item()

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            if i % self.log_freq == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))
                if self.use_wandb:
                    wandb.log(post_fix)
                print(
                    "EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / (i + 1) 
                )

        if not train:
            decoded = self.eval_sample()
            print(decoded)
            if self.use_wandb:
                self.table_rows.append([epoch, avg_loss / (i+1), decoded])
                table = wandb.Table(data=self.table_rows, columns=["epoch", "avg_loss", "sample"])
                wandb.log({"samples": table})

    def eval_sample(self):
        prompt = random.choice(sample_prompts)
        tokenized = self.tokenizer(
            prompt, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        eval_batch = self.collator([tokenized])
        input_ids = eval_batch["input_ids"].squeeze(0).to(self.device)
        mask_lm_output = self.model.forward(input_ids)
        output = torch.argmax(mask_lm_output, dim=2)
        decoded = self.tokenizer.decode(output[0])
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
