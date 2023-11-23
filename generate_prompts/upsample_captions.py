import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import pipeline, Pipeline
import fire
from huggingface_hub import login
from dotenv import load_dotenv
from vllm import LLM, SamplingParams, RequestOutput


def load_chat_pipeline_hf():
    """Loads the HuggingFaceH4/zephyr-7b-alpha model and wraps into a handy text-generation pipeline."""
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-alpha",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe


def get_messages_for_chat() -> Tuple[Dict, List[Dict]]:
    """
    Prepares the system and user-assistant style messages for inference.

    Example messages come from the DALL-E 3 technical report:
    https://cdn.openai.com/papers/dall-e-3.pdf.
    """
    system_message = {
        "role": "system",
        "content": """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

    There are a few rules to follow:

    - You will only ever output a single image description per user request.
    - Sometimes the user will request that you modify previous captions. In this case, you should refer to your previous conversations with the user and make the modifications requested.
    - When modifications are requested, you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.
    - Other times the user will not want modifications, but instead want a new image. In this case, you should ignore your previous conversation with the user."
    - Image descriptions must be between 15-80 words. Extra words will be ignored.
    """,
    }

    user_conversation = [
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'make the light red'",
        },
        {
            "role": "assistant",
            "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head. the blade glows with a red light, casting a warm glow on the trees and bushes surrounding him.",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'draw a frog playing dominoes'",
        },
        {
            "role": "assistant",
            "content": "a frog sits on a worn table playing a game of dominoes with an elderly raccoon. the table is covered in a green cloth, and the frog is wearing a jacket and a pair of jeans. The scene is set in a forest, with a large tree in the background.",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : '{prompt}'",
        },
    ]
    return system_message, user_conversation


def upsample_caption_hf(pipeline: Pipeline, message: list[Dict[str, str]]):
    """Performs inference on a single prompt."""
    outputs = pipeline(
        message,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs


def upload_dataset(
    hf_dataset: Dataset, hf_dataset_name: str, new_dataset_rows: List[Dict]
):
    dataset_new_rows = Dataset.from_list(new_dataset_rows)
    dataset_new_rows.to_csv("upsampled_new_prompts.csv")

    concat_dataset = concatenate_datasets([hf_dataset, dataset_new_rows])

    print(f"Uploading {len(new_dataset_rows)} new prompts to the Hub...")
    concat_dataset.push_to_hub(hf_dataset_name)


def main():
    hf_dataset_name = "roborovski/upsampled-prompts-parti"

    print("Loading existing prompts...")
    hf_dataset: Dataset = load_dataset(hf_dataset_name, split="train")  # type: ignore

    print("Loading new prompts...")
    parti_prompts: pd.DataFrame = pd.read_csv("PartiPrompts.tsv", sep="\t")

    source_prompts_list = parti_prompts

    new_dataset_rows: List[Dict] = []

    print("Logging into the Hub...")
    file_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(file_dir, ".env"))
    token = os.getenv("HF_TOKEN")
    print(f"Logging in with token: {token}")
    login(token=token, add_to_git_credential=True)

    # initial test upload before loading the pipeline
    upload_dataset(hf_dataset, hf_dataset_name, new_dataset_rows)

    n_epochs = 100

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
    print("Loading local pipeline...")
    model = LLM(model="HuggingFaceH4/zephyr-7b-beta", dtype="auto")
    print("Pipeline loaded.")

    tokenizer = model.get_tokenizer()

    print("Upsampling captions...")
    for epoch in range(n_epochs):
        for i, row in enumerate(source_prompts_list.itertuples()):
            original_prompt, category = row.Prompt, row.Category
            system_message, user_conversation = get_messages_for_chat()
            updated_prompt = user_conversation[-1]["content"].format(
                prompt=original_prompt
            )
            user_conversation[-1]["content"] = updated_prompt

            final_message = [system_message, *user_conversation]
            full_conversation_formatted: str = tokenizer.apply_chat_template(  # type: ignore
                final_message, tokenize=False, add_generation_prompt=True
            )

            outputs = model.generate(full_conversation_formatted, sampling_params)

            upsampled_caption = outputs[0].outputs[0].text
            new_dataset_rows.append(
                {
                    "Prompt": original_prompt,
                    "Category": category,
                    "Upsampled": upsampled_caption,
                }
            )

            print(
                f"Upsampled prompt {epoch} {i} ({category}): {original_prompt} -> {upsampled_caption}"
            )

            if i % 500 == 0:
                print(f"Upsampled {i} prompts")
                upload_dataset(hf_dataset, hf_dataset_name, new_dataset_rows)

        upload_dataset(hf_dataset, hf_dataset_name, new_dataset_rows)


if __name__ == "__main__":
    fire.Fire(main)
