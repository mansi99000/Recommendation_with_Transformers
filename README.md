# Recommendation with Transformers

A **multimodal** exploration of Transformers for personalized movie recommendations, combining natural-language queries with sequential and collaborative signals on the MovieLens dataset.

## Multimodal setup

This project is **multimodal**: it uses two kinds of input to produce recommendations.

1. **Text** — Natural-language queries (e.g. *"something like Inception"*) and movie metadata (titles) are encoded with a sentence encoder and used to match the user’s request to movies.
2. **Sequential / collaborative** — A Transformer encoder uses **user IDs** and **sequences of watched movies** to model behavior and preferences.

Recommendations are produced by fusing text-based similarity with the Transformer’s scores, and the core pipeline is evaluated with causal masking and stronger baselines (Markov chain, SVD, popularity).

## What’s in the repo

- **`recommendation.py`** — Main pipeline: train the Transformer (with causal masking), compare against Markov chain, SVD collaborative filtering, and popularity baselines; save checkpoints and results to `checkpoints/` and `results/`.
- **`conversational_recommend.py`** — Conversational entry point: type a natural-language query and get top-K movie recommendations (fuses query–title similarity with the trained Transformer).
- **`ablation_study.py`** — Ablation over embedding size, layers, heads, and sequence length; writes `results/ablation_results_*.csv`.
- **`transformer.py`** & **`positional_encoder.py`** — Reusable Transformer and positional encoding used by the main script.

## Dataset

**MovieLens 1M** — ~1M ratings from ~6,000 users on ~3,900 movies.

- `ratings.dat` — UserID::MovieID::Rating::Timestamp  
- `users.dat` — UserID::Gender::Age::Occupation::Zip-code  
- `movies.dat` — MovieID::Title::Genres  

Data is downloaded automatically if `ml-1m/` is missing.

## How to run

```bash
# Setup
python3 -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train and evaluate (Transformer + baselines), save results
python3 recommendation.py

# Ask for recommendations in natural language (multimodal: text + model)
python3 conversational_recommend.py "something like Inception"
python conversational_recommend.py   # interactive prompt

# Optional: ablation study (writes results/ablation_results_*.csv)
python3 ablation_study.py
```

## Results

After running `recommendation.py`, metrics (NDCG@K, Precision@K, Hit Rate@K) for the Transformer and baselines are printed and saved under `results/recommendation_results_<timestamp>.csv`. The Transformer is evaluated with causal masking and outperforms the Markov, SVD, and popularity baselines.
