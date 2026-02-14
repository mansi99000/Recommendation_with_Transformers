#!/usr/bin/env python3
"""
Conversational entry point: recommend movies from a natural-language query.

Uses (1) a sentence encoder to match your query to movie titles and (2) the
trained Transformer to score movies. Fuses both and returns top-K recommendations.

Usage:
  python conversational_recommend.py "something like Inception"
  python conversational_recommend.py "funny romantic comedy"
  python conversational_recommend.py   # interactive: prompts for query
"""

import os
import sys
import argparse

import torch

# Same vocab and model as main pipeline — single source of truth
from transformer import TransformerModel
from data_utils import load_movielens_data

# -----------------------------------------------------------------------------
# Data loading (must match recommendation.py so vocabs align with checkpoint)
# -----------------------------------------------------------------------------

def load_vocab_and_titles():
    """Load MovieLens 1M and build vocab + movie_id -> title mapping.

    Downloads the dataset automatically if not already present.
    """
    data = load_movielens_data()
    return data["movie_vocab"], data["user_vocab"], data["movie_title_dict"]


# -----------------------------------------------------------------------------
# Model and sentence encoder
# -----------------------------------------------------------------------------

def load_model(checkpoint_path, movie_vocab, user_vocab, device):
    """Load trained Transformer from checkpoint."""
    ntokens = len(movie_vocab)
    nusers = len(user_vocab)
    emsize = 128
    d_hid = 128
    nlayers = 2
    nhead = 2
    dropout = 0.2

    model = TransformerModel(
        ntokens, nusers, emsize, nhead, d_hid, nlayers, dropout
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def get_sentence_encoder():
    """Lazy load sentence transformer (first call may download)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install: pip install sentence-transformers")
        sys.exit(1)
    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------------------------------------------
# Scoring: content (query vs titles) + model (Transformer with default context)
# -----------------------------------------------------------------------------

def encode_movie_titles(movie_vocab, movie_title_dict, encoder):
    """Encode each movie title in vocab order. Returns (n_movies, dim)."""
    itos = movie_vocab.get_itos()
    titles = []
    for mid in itos:
        if mid == "<unk>":
            titles.append("Unknown")
        else:
            titles.append(movie_title_dict.get(mid, mid))
    return encoder.encode(titles, show_progress_bar=False)


def recommend(
    query: str,
    model,
    movie_vocab,
    user_vocab,
    movie_title_dict,
    encoder,
    device,
    top_k: int = 10,
    alpha: float = 0.5,
):
    """
    Fuse query-based (content) scores and Transformer scores; return top-K titles.

    alpha: weight for content (query vs titles). 1-alpha = weight for model.
    """
    import numpy as np

    ntokens = len(movie_vocab)
    itos = movie_vocab.get_itos()
    unk_id = movie_vocab.get_stoi()["<unk>"]

    # ---- Content branch: query vs movie titles ----
    query_embed = encoder.encode([query], show_progress_bar=False)
    movie_embeds = encode_movie_titles(movie_vocab, movie_title_dict, encoder)

    query_embed = query_embed.astype(np.float32)
    movie_embeds = movie_embeds.astype(np.float32)
    # Cosine similarity (assume L2-normalized; sentence-transformers often are)
    q_norm = query_embed / (np.linalg.norm(query_embed, axis=1, keepdims=True) + 1e-9)
    m_norm = movie_embeds / (np.linalg.norm(movie_embeds, axis=1, keepdims=True) + 1e-9)
    content_scores = np.dot(m_norm, q_norm.T).flatten()

    # ---- Model branch: Transformer with "no history" (padding) ----
    seq_len = 3  # same as recommendation.py (input is sequence[:-1])
    default_seq = torch.full((1, seq_len), unk_id, dtype=torch.long, device=device)
    default_user = torch.zeros(1, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(default_seq, default_user)
    model_scores = logits[0, -1, :].cpu().numpy()

    # ---- Fuse: normalize to [0,1] then combine ----
    def minmax(x):
        x = np.asarray(x, dtype=np.float64)
        mi, ma = x.min(), x.max()
        if ma - mi < 1e-9:
            return np.ones_like(x)
        return (x - mi) / (ma - mi)

    content_n = minmax(content_scores)
    model_n = minmax(model_scores)
    final_scores = alpha * content_n + (1 - alpha) * model_n

    top_indices = np.argsort(final_scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        mid = itos[idx]
        title = movie_title_dict.get(mid, mid) if mid != "<unk>" else "Unknown"
        results.append((title, float(final_scores[idx])))
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Conversational movie recommendations")
    parser.add_argument("query", nargs="*", help="Query in natural language (e.g. 'something like Inception')")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for query vs model (0=model only, 1=query only)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model_params.pt", help="Model checkpoint")
    args = parser.parse_args()

    if args.query:
        query = " ".join(args.query)
    else:
        query = input("Enter your request (e.g. 'something like Inception'): ").strip()
    if not query:
        query = "good movies"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading vocab and titles...")
    movie_vocab, user_vocab, movie_title_dict = load_vocab_and_titles()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run recommendation.py first to train and save the model.")
        sys.exit(1)
    print("Loading model...")
    model = load_model(args.checkpoint, movie_vocab, user_vocab, device)

    print("Loading sentence encoder (first run may download)...")
    encoder = get_sentence_encoder()

    print(f"\nQuery: \"{query}\"\nTop-{args.top_k} recommendations:\n")
    results = recommend(
        query, model, movie_vocab, user_vocab, movie_title_dict,
        encoder, device, top_k=args.top_k, alpha=args.alpha,
    )
    for i, (title, score) in enumerate(results, 1):
        print(f"  {i:2}. {title}")

    print()


if __name__ == "__main__":
    main()
