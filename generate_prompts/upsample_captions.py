import os
from pprint import pprint
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import pipeline, Pipeline
import fire
from huggingface_hub import login, whoami
from dotenv import load_dotenv

def load_chat_pipeline():
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

    rest_of_the_message = [
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: 'a man holding a sword'",
        },
        {
            "role": "assistant",
            "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head. the blade glows with a blue light , casting a soft glow on the trees and bushes surrounding him.",
        },
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
    return system_message, rest_of_the_message


def make_final_message(
    system_message: Dict[str, str],
    rest_of_the_message: List[Dict[str, str]],
    debug=False,
):
    """Prepares the final message for inference."""
    final_message = [system_message]
    final_message.extend(rest_of_the_message)
    if debug:
        pprint(final_message)
    return final_message


def upsample_caption_local(pipeline : Pipeline, message: list[Dict[str, str]]):
    """Performs inference on a single prompt."""
    prompt = pipeline.tokenizer.apply_chat_template( # type: ignore
        message, tokenize=False, add_generation_prompt=True
    )
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs


def upsample_caption_oai(message: list[Dict[str, str]]):
    pass


def prepare_assistant_reply(assistant_output):
    """Prepares the assistant reply which will be considered as the upsampled caption."""
    output = assistant_output[0]["generated_text"]
    parts = output.rsplit("<|assistant|>", 1)
    assistant_reply = parts[1].strip() if len(parts) > 1 else None
    return assistant_reply


def upload_dataset(hf_dataset: Dataset, hf_dataset_name: str, new_dataset_rows: List[Dict]):

    dataset_new_rows = Dataset.from_list(new_dataset_rows)
    dataset_new_rows.to_csv("upsampled_new_prompts.csv")

    concat_dataset = concatenate_datasets([hf_dataset, dataset_new_rows])

    print(f"Uploading {len(new_dataset_rows)} new prompts to the Hub...")
    concat_dataset.push_to_hub(hf_dataset_name)


def main(local: bool = False):

    if not local:
        raise NotImplementedError("Only local mode is supported for now.")

    hf_dataset_name = "roborovski/upsampled-prompts-parti"

    print("Loading existing prompts...")
    hf_dataset: Dataset = load_dataset(hf_dataset_name, split="train") # type: ignore

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

    pipeline = None
    if local:
        print("Loading local pipeline...")
        pipeline = load_chat_pipeline()

    print("Upsampling captions...")
    for epoch in range(n_epochs):
        for i, row in enumerate(source_prompts_list.itertuples()):
            prompt, category = row.Prompt, row.Category
            system_message, rest_of_the_message = get_messages_for_chat()
            updated_prompt = rest_of_the_message[-1]["content"].format(prompt=prompt)
            rest_of_the_message[-1]["content"] = updated_prompt
            final_message = make_final_message(
                system_message, rest_of_the_message, debug=False
            )

            if local:
                outputs = upsample_caption_local(pipeline, final_message) # type: ignore
            else:
                outputs = upsample_caption_oai(final_message)

            upsampled_caption = prepare_assistant_reply(outputs)
            new_dataset_rows.append({"Prompt": prompt, "Category": category, "Upsampled": upsampled_caption})

            print(f"Upsampled prompt {epoch} {i} ({category}): {prompt} -> {upsampled_caption}")

            if i % 500 == 0:
                print(f"Upsampled {i} prompts")
                upload_dataset(hf_dataset, hf_dataset_name, new_dataset_rows)

        upload_dataset(hf_dataset, hf_dataset_name, new_dataset_rows)


if __name__ == "__main__":
    fire.Fire(main)
