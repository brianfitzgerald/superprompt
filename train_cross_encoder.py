from typing import Dict
from datasets import load_dataset
import spacy
import wandb
from models.clip_emb_aug import CLIPEmbeddingAugmenter
from models.cross_encoder import CrossEncoder, multiple_negatives_ranking_loss
import fire
import torch
import wandb
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchinfo import summary
from utils import get_available_device, get_model_gradient_norm
from typing import Dict
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import ndcg_score, recall_score, precision_score
from dataclasses import dataclass
from torch import Tensor
import torch.nn.functional as F

if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")

torch.manual_seed(0)


@dataclass
class LossFnOutput:
    loss: Tensor
    subject_out: Tensor
    descriptor_out: Tensor
    scores: Tensor
    labels: Tensor


def loss_fn_emb_aug(
    batch: Dict,
    model: CLIPEmbeddingAugmenter,
) -> LossFnOutput:
    out = {}
    for key in ("subject", "descriptor"):
        emb = model(batch[key])
        out[key] = emb
    # get the embeddings for both the subject and the descriptor batch
    scores, labels = multiple_negatives_ranking_loss(out["subject"], out["descriptor"])
    loss = F.cross_entropy(scores, labels)
    return LossFnOutput(loss, out["subject"], out["descriptor"], scores, labels)


# eval_every and valid_every are in terms of batches
def main(use_wandb: bool = False, eval_every: int = 10, valid_every: int = 100):
    device = get_available_device()

    print("Loading dataset..")
    dataset = load_dataset("roborovski/diffusiondb-seq2seq")

    print("Loading model..")
    model = CrossEncoder(device)
    print(summary(model))

    # Hyperparameters
    num_epochs: int = 200
    learning_rate: float = 2e-5
    batch_size: int = 64
    warmup_steps: int = 10
    max_grad_norm = 1

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    steps_per_epoch = len(dataset["train"]) // batch_size
    num_training_steps = steps_per_epoch * num_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )

    if use_wandb:
        wandb.init(project="superprompt-cross-encoder")
        wandb.watch(model)

    dataset = dataset["train"].train_test_split(test_size=int(48), seed=42)
    train_loader = DataLoader(dataset["train"], batch_size=batch_size)
    eval_loader = DataLoader(dataset["test"], batch_size=len(dataset["test"]))

    for i, epoch in enumerate(range(num_epochs)):
        train_iter = tqdm(train_loader, total=len(train_loader))
        for batch in train_iter:
            out = loss_fn_emb_aug(batch, model)
            lr = optimizer.param_groups[0]["lr"]

            loss_formatted = round(out.loss.item(), 4)
            lr_formatted = round(lr, 8)
            gradient_norm = round(get_model_gradient_norm(model), 4)
            log_dict = {
                "loss": loss_formatted,
                "lr": lr_formatted,
                "grad_norm": gradient_norm,
                "epoch": epoch,
            }

            train_iter.set_postfix(log=log_dict)
            if use_wandb:
                wandb.log(log_dict)

            # Backward pass and optimization
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if i % eval_every == 0:
                print("---Running eval---")
                eval_iter = tqdm(eval_loader, total=len(eval_loader))
                model.eval()
                for batch in eval_iter:
                    out = loss_fn_emb_aug(batch, model)
                    labels = out.labels.cpu().detach().numpy()
                    scores = torch.argmax(out.scores, dim=1)
                    scores = scores.cpu().detach().numpy()

                    # Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
                    # sentence_transformers\evaluation\InformationRetrievalEvaluator.py

                    loss_formatted = round(out.loss.item(), 4)
                    ndcg = ndcg_score(labels, scores)
                    recall = recall_score(labels, scores)
                    precision_score = precision_score(labels, scores)
                    f1_score = (
                        2 * (precision_score * recall) / (precision_score + recall)
                    )

                    log_dict = {
                        "eval_loss": loss_formatted,
                        "ndcg": ndcg,
                        "recall": recall,
                        "precision": precision_score,
                        "f1": f1_score,
                    }

                    if use_wandb:
                        wandb.log(log_dict)

                    print("---Eval stats---")
                    print(log_dict)

                model.train()

        # # TODO rewrite this. use the retrieved annotations to generate images
        # if i % valid_every == 0:
        #     eval_iter = tqdm(eval_loader, total=len(eval_loader))
        #     pipe = StableDiffusionPipeline.from_pretrained(
        #         "runwayml/stable-diffusion-v1-5",
        #         torch_dtype=torch.float16,
        #         safety_checker=None,
        #     )
        #     pipe = pipe.to("cuda")
        #     if is_xformers_available():
        #         pipe.unet.enable_xformers_memory_efficient_attention()
        #     for batch in eval_iter:

        #         loss = loss_fn_emb_aug(batch, model)

        #         log_dict = {
        #             "loss": loss.item(),
        #             "unmasked": [],
        #             "encoded": [],
        #         }

        #         # clear out directory
        #         Path("out").mkdir(exist_ok=True)
        #         shutil.rmtree("out")
        #         Path("out").mkdir(exist_ok=True)

        #         generations = pipe(prompt="").images
        #         os.makedirs("out", exist_ok=True)
        #         for i, generation in enumerate(generations):
        #             generation.save(f"out/{key}_{i}.png")
        #             if use_wandb:
        #                 log_dict[key].append(wandb.Image(generation))

        #         if use_wandb:
        #             wandb.log(log_dict)

        #     del pipe
        #     gc.collect()
        #     torch.cuda.empty_cache()

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
