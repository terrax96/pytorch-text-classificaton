# pytorch-text-classificaton

Text classification of the AG News dataset. Reference tutorial can be found [here](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html).

## Setup

- `python -m venv venv`
- Linux: `source venv/bin/activate` OR Windows: `Set-ExecutionPolicy Unrestricted -Scope Process; .\venv\Scripts\Activate.ps1`
- `pip install -r requirements.txt`

## Run

`python src/main.py`

## Inspect trained models

`tensorboard --logdir lightning_logs`