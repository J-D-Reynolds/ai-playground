# AI Playground

A machine learning testing project featuring various NLP and computer vision tasks using Hugging Face Transformers.

## Features

- **Sentiment Analysis** - Classify text sentiments with DistilBERT
- **Image Processing** - Download and process images from URLs
- **Text Generation** - Generate text using DistilGPT-2
- **Named Entity Recognition (NER)** - Extract named entities from text
- **Model Testing Utilities** - Benchmark and test ML pipelines

## Setup

Prerequisites: Python 3.12+, [uv](https://github.com/astral-sh/uv)

Install dependencies:
```bash
uv sync
```

## Running

Execute all ML tests:
```bash
uv run python main.py
```

## Project Structure

- `main.py` - Main testing script with 5 ML test cases
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Locked dependency versions

## Dependencies

- `torch` - Deep learning framework
- `transformers` - Hugging Face transformers library
- `requests` - HTTP client
- `Pillow` - Image processing

## Requirements

- torch >= 2.11.0
- transformers >= 5.5.3
