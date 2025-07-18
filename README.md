# photorank

A fun little script to pick the "best" image from a folder of images, using a local LLM with vision capabilities as the judge.

## Features:
- Tournament-style elimination rounds for image ranking.
- AI-powered judging using DSPy and Ollama.
- Decision and score logging for auditability.
- Privacy focused - runs entirely locally, your data never leaves your system.

## Prerequisites

- [Ollama](https://ollama.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

## Quick Start
```
# pull the default vision model
ollama pull gemma3:4b

# clone and run
git clone https://github.com/jeffgreenca/photorank 
cd photorank
uv run photorank.py <path/to/images>
```

> `uv` will automatically create a Python virtual env and install the required dependnecies.

## Usage

### Model Selection

Photorank can work with any [Ollama vision model](https://ollama.com/search?c=vision) that will run on your system. For this example, we choose `qwen2.5vl:3b`, a fast 3.2 GB model.

```
ollama pull qwen2.5vl:3b
uv run photorank.py --model qwen2.5vl:3b <path/to/images>
```

### Other Configuration

Check the built-in help:

```
uv run photorank.py --help
```

### License

This project is licensed under the [MIT License](LICENSE).