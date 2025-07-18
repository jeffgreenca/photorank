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

# DSPy will generate a prompt from this class, and ensure the LLM's output satisfies the type annotations.
class Judge(dspy.Signature):
    first_image: dspy.Image = dspy.InputField()
    second_image: dspy.Image = dspy.InputField()
    first_image_evaluation: str = dspy.OutputField()
    second_image_evaluation: str = dspy.OutputField()
    judges_comments: str = dspy.OutputField()
    choice: Literal["1", "2"] = dspy.OutputField(desc=
                                                 ("Consider the first image as choice 1 and the second image as choice 2. The choice must be either '1' or '2'.\n"
                                                  "When making a choice, consider the overall quality, composition, interest, sharpness, clarity, lighting and appeal of the images, "
                                                  "and choose the one that you think is better."))

class TournamentLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log_decision(self, files, winner, response):
        """Append the decision to the log file."""
        with open(self.log_file, "a") as f:
            entry = {
                "timestamp": time.time() / 1000,
                "files": files,
                "winner": winner,
            }
            # Dynamically add all output fields from Judge signature
            for field in Judge.output_fields:
                entry[field] = getattr(response, field, None)
            f.write(json.dumps(entry) + "\n")

    def log_scores(self, scores: dict[str, int]):
        """Log the current scores to the log file."""
        with open(self.log_file, "a") as f:
            entry = {
                "timestamp": time.time() / 1000,
                "scores": sorted(scores.items(), key=lambda x: x[1], reverse=True),
            }
            f.write(json.dumps(entry) + "\n")

class ImageProcessor:
    def __init__(self, max_side=900):
        self.max_side = max_side

    @lru_cache(maxsize=None)
    def process(self, file: str) -> dspy.Image:
        """Resize and ensure the image is in RGB format."""
        im = Image.open(file)
        im.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
        im = im.convert("RGB")
        return dspy.Image.from_PIL(im)


class PhotoRanker:
    def __init__(self, image_processor: ImageProcessor, tournament_logger: TournamentLogger, judge: dspy.Signature, module: dspy.Module):
        self.image_processor = image_processor
        self.tournament_logger = tournament_logger

        if 'first_image' not in judge.input_fields or 'second_image' not in judge.input_fields:
            raise ValueError("Judge signature must have 'first_image' and 'second_image' fields.")
        if 'choice' not in judge.output_fields:
            raise ValueError("Judge signature must have 'choice' field.")

        self.judge = module(judge)

    def compare(self, entries):
        """Use the DSPy defined judge to compare two images, returning the winner."""
        if len(entries) != 2:
            raise ValueError("compare function expects exactly two entries.")

        images = list(map(self.image_processor.process, entries))

        r = self.judge(first_image=images[0], second_image=images[1])
        choice_num = int(r.choice)
        if choice_num not in [1, 2]:
            raise ValueError(f"Invalid choice {r.choice}, expected '1' or '2'.")
        winner = entries[choice_num - 1]
        self.tournament_logger.log_decision(entries, winner, r)
        return winner


    def round(self, entries):
        """Conduct a single elimination round, comparing pairs of images and returning the winners."""
        winners = []
        for i in tqdm(range(0, len(entries), 2)):
            if i + 1 < len(entries):
                winner = self.compare(entries[i:i + 2])
                if winner is None:
                    raise ValueError("Comparison returned None, which is unexpected.")
                winners.append(winner)
            else:
                winners.append(entries[i])
                logger.info(
                    f"Odd number of candidates, {entries[i]} goes through without comparison."
                )
        return winners

    def tournament(self, entries: list[str]) -> str | None:
        """Conduct an elimination tournament on the image file entries, returning the final winner."""
        rand = random.Random(42)  # Fixed seed for reproducibility

        score = Counter()

        while len(entries) > 1:
            # random pairings
            rand.shuffle(entries)

            entries = self.round(entries)
            # Update scores based on the number of wins
            for entry in entries:
                score[entry] += 1
            logger.info(f"Round complete, {len(entries)} entries remaining.")

            # log current scores both to console and to file
            logger.info(
                f"Current scores: {sorted(score.items(), key=lambda x: x[1], reverse=True)}"
            )
            self.tournament_logger.log_scores(score)

        return entries[0] if entries else None

def get_initial_files(path, filetypes):
    """Collect all image files in the specified path with the given filetypes."""
    filetypes = filetypes.split(",")
    filetypes = [ft.strip() for ft in filetypes if ft.strip()]
    if not filetypes:
        raise ValueError("No filetypes provided.")
    entries = []
    for ext in filetypes:
        entries.extend(glob(os.path.join(path, f"*.{ext}")))
    logger.info(f"Found {len(entries)} files in {path} with extensions {filetypes}.")
    return entries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a photo ranking tournament.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, help="Path to the images to rank.")
    parser.add_argument("--size", type=int, default=900, help="Images are resized to fit within this size (longest side, preserving aspect ratio). Smaller is faster.")
    parser.add_argument("--filetypes", type=str, default="jpg,jpeg,png,JPG,JPEG,PNG",
                        help="Comma-separated list of image file extensions to include in the tournament. Case sensitive.")
    parser.add_argument("--model", type=str, default="gemma3:4b", help="Ollama vision model to use for judging.")
    parser.add_argument("--module", type=str, default="ChainOfThought", choices=["ChainOfThought", "Predict"],
                        help="DSPy module to use as the judge. Predict is the fastest, but ChainOfThought is probably more accurate.")
    parser.add_argument("--decision-log", type=str, default="decisions.jsonl", help="File to log decisions.")

    args = parser.parse_args()

    api_base = os.getenv("PHOTORANK_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    api_key = os.getenv("PHOTORANK_API_KEY", "")

    model = args.model

    if "/" in model and not model.startswith("openai/"):
        raise ValueError(
            "Ollama model must start with 'openai/' - do not use ollama or ollama_chat, they are not compatible with DSPy image sending."
        )

    # bare model, add prefix
    if not model.startswith("openai/"):
        model = "openai/" + model

    # force DSPy to use Ollama's OpenAI compatibility API, not the default ollama API.
    # see https://github.com/stanfordnlp/dspy/issues/8067#issuecomment-3084817296
    if not api_base.endswith("/v1"):
        api_base += "/v1"

    # global configuration
    lm = dspy.LM(model, api_base=api_base, api_key=api_key)
    dspy.configure(lm=lm)

    # we can fail fast if the user specified an invalid model by a quick test
    lm("Ping - please respond with 'pong'.")

    if args.module == "ChainOfThought":
        module = dspy.ChainOfThought
    elif args.module == "Predict":
        module = dspy.Predict
    else:
        raise ValueError(f"Unknown module type: {args.module}.")

    tourny = PhotoRanker(
        image_processor=ImageProcessor(max_side=args.size),
        tournament_logger=TournamentLogger(log_file=args.decision_log),
        judge=Judge,
        module=module,
    )

    entries = get_initial_files(args.path, args.filetypes)
    if len(entries) < 2:
        raise ValueError("Not enough images to conduct a tournament. At least two images are required.")

    winner = tourny.tournament(entries)
    print(winner)
