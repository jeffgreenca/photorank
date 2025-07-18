#!/usr/bin/env python3
import os
import time
import json
import logging
import argparse
import random
from glob import glob
from collections import Counter
from functools import lru_cache

from PIL import Image
from tqdm import tqdm
from typing import Literal

import dspy

from dotenv import load_dotenv

load_dotenv()

def configure_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

logger = configure_logging()

# env var configurations
api_base = os.getenv("PHOTORANK_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
api_key = os.getenv("PHOTORANK_API_KEY", "")
ollama_model = os.getenv("PHOTORANK_OLLAMA_MODEL", "openai/gemma3:12b")
log_file = os.getenv("PHOTORANK_DECISION_LOG_FILE", "photorank-decisions.jsonl")
default_max_side = int(
    os.getenv("PHOTORANK_MAX_SIDE", 600)
)  # Rescale images to fit within this size (longest side, preserving aspect ratio). Smaller is faster.

if "/" in ollama_model and not ollama_model.startswith("openai/"):
    raise ValueError(
        "Ollama model must start with 'openai/' - do not use ollama or ollama_chat, they are not compatible with DSPy image sending."
    )

# bare model, add prefix
if not ollama_model.startswith("openai/"):
    ollama_model = "openai/" + ollama_model

# force DSPy to use Ollama's OpenAI compatibility API, not the default ollama API.
# see https://github.com/stanfordnlp/dspy/issues/8067#issuecomment-3084817296
if not api_base.endswith("/v1"):
    api_base += "/v1"

# Configure AI judge
lm = dspy.LM(ollama_model, api_base=api_base, api_key=api_key)
dspy.configure(lm=lm)


# DSPy will generate a prompt from this class, and ensure the LLM's output satisfies the type annotations.
class Judge(dspy.Signature):
    first_image: dspy.Image = dspy.InputField()
    second_image: dspy.Image = dspy.InputField()
    first_image_evaluation: str = dspy.OutputField()
    second_image_evaluation: str = dspy.OutputField()
    judges_comments: str = dspy.OutputField()
    choice: Literal["1", "2"] = dspy.OutputField()


# "judge" now holds a callable that can be used to compare two images.
# It will return an object with the fields defined in the Judge class.
judge = dspy.Predict(Judge)


def log_decision(files, winner, response):
    """Append the decision to the log file."""
    with open(log_file, "a") as f:
        entry = {
            "timestamp": time.time() / 1000,
            "files": files,
            "winner": winner,
            "first_image_evaluation": response.first_image_evaluation,
            "second_image_evaluation": response.second_image_evaluation,
            "judges_comments": response.judges_comments,
            "choice": response.choice,
            "model": ollama_model,
        }
        f.write(json.dumps(entry) + "\n")


def log_scores(scores):
    """Log the current scores to the log file."""
    with open(log_file, "a") as f:
        entry = {
            "timestamp": time.time() / 1000,
            "scores": sorted(scores.items(), key=lambda x: x[1], reverse=True),
            "model": ollama_model,
        }
        f.write(json.dumps(entry) + "\n")


@lru_cache
def prepare_image(file, max_side=default_max_side):
    """Resize and ensure the image is in RGB format."""
    im = Image.open(file)
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    im = im.convert("RGB")
    return dspy.Image.from_PIL(im)


def compare(entries):
    """Use the DSPy defined judge to compare two images, returning the winner."""
    if len(entries) != 2:
        raise ValueError("compare function expects exactly two entries.")

    images = list(map(prepare_image, entries))

    r = judge(first_image=images[0], second_image=images[1])
    choice_num = int(r.choice)
    if choice_num not in [1, 2]:
        raise ValueError(f"Invalid choice {r.choice}, expected '1' or '2'.")
    winner = entries[choice_num - 1]
    log_decision(entries, winner, r)
    return winner


def round(entries):
    """Conduct a single elimination round, comparing pairs of images and returning the winners."""
    winners = []
    for i in tqdm(range(0, len(entries), 2)):
        if i + 1 < len(entries):
            winner = compare(entries[i:i + 2])
            if winner is None:
                raise ValueError("Comparison returned None, which is unexpected.")
            winners.append(winner)
        else:
            winners.append(entries[i])
            logger.info(
                f"Odd number of candidates, {entries[i]} goes through without comparison."
            )
    return winners

def get_initial_files(path, filetypes):
    """Collect all image files in the specified path with the given filetypes."""
    filetypes = filetypes.split(",")
    filetypes = [ft.strip() for ft in filetypes if ft.strip()]
    if not filetypes:
        raise ValueError("No filetypes provided.")
    entries = []
    for ext in filetypes:
        entries.extend(glob(os.path.join(path, f"*.{ext}")))
    if not entries:
        raise ValueError(f"No files found in {path} with extensions {filetypes}.")
    logger.info(f"Found {len(entries)} files in {path} with extensions {filetypes}.")
    return entries

def tournament(path, filetypes):
    """Conduct a tournament on the images found in the specified path, returning a single final winner, the best image."""
    entries = get_initial_files(path, filetypes)

    rand = random.Random(42)  # Fixed seed for reproducibility

    score = Counter()

    while len(entries) > 1:
        # random pairings
        rand.shuffle(entries)

        entries = round(entries)
        # Update scores based on the number of wins
        for entry in entries:
            score[entry] += 1
        logger.info(f"Round complete, {len(entries)} entries remaining.")

        # log current scores both to console and to file
        logger.info(
            f"Current scores: {sorted(score.items(), key=lambda x: x[1], reverse=True)}"
        )
        log_scores(score)

    return entries[0] if entries else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a photo ranking tournament.")
    parser.add_argument("path", type=str, help="Path to the images to rank.")
    parser.add_argument("--filetypes", type=str, default="jpg,jpeg,png,JPG,JPEG,PNG",
                        help="Comma-separated list of image file extensions to include in the tournament. Case sensitive.")
    args = parser.parse_args()

    winner = tournament(args.path, args.filetypes)
    print(winner)
