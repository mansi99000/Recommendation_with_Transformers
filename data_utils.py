"""
Shared data utilities for the Transformer-based movie recommendation pipeline.

Provides vocabulary construction, dataset classes, sequence windowing, and
data loading helpers used by recommendation.py, ablation_study.py, and
conversational_recommend.py.
"""

import os
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from urllib.request import urlretrieve

import pandas as pd
import numpy as np


# ============================================================
# Vocabulary
# ============================================================

class SimpleVocab:
    """Lightweight replacement for torchtext.vocab.vocab (which is deprecated)."""

    def __init__(self, counter, specials=None):
        specials = specials or []
        self.itos = list(specials) + [tok for tok, _ in counter.most_common() if tok not in specials]
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos

    def __len__(self):
        return len(self.itos)


# ============================================================
# Dataset
# ============================================================

class MovieSeqDataset(Dataset):
    """PyTorch Dataset for user movie-interaction sequences."""

    def __init__(self, data, movie_vocab_stoi, user_vocab_stoi):
        self.data = data
        self.movie_vocab_stoi = movie_vocab_stoi
        self.user_vocab_stoi = user_vocab_stoi

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, movie_sequence = self.data[idx]
        movie_data = [self.movie_vocab_stoi[item] for item in movie_sequence]
        user_data = self.user_vocab_stoi[user]
        return torch.tensor(movie_data), torch.tensor(user_data)


# ============================================================
# Sequence Windowing
# ============================================================

def create_sequences(values, window_size, step_size, min_history=1):
    """Create sliding-window sequences from a list of movie IDs."""
    sequences = []
    start_index = 0
    while len(values[start_index:]) > min_history:
        seq = values[start_index : start_index + window_size]
        sequences.append(seq)
        start_index += step_size
    return sequences


# ============================================================
# Collate Function Factory
# ============================================================

def make_collate_fn(unk_idx):
    """Create a collate function for DataLoader with the given padding index."""
    def collate_batch(batch):
        movie_list = [item[0] for item in batch]
        user_list = [item[1] for item in batch]
        return (
            pad_sequence(movie_list, padding_value=unk_idx, batch_first=True),
            torch.stack(user_list),
        )
    return collate_batch


# ============================================================
# Data Loading
# ============================================================

def load_movielens_data():
    """Load and preprocess MovieLens 1M dataset.

    Downloads the dataset if not already present. Returns a dictionary containing
    the raw DataFrames, vocabulary objects, and grouped ratings data.

    Returns:
        dict with keys:
            users, ratings, movies: raw DataFrames
            movie_vocab, movie_vocab_stoi: movie vocabulary and string-to-index mapping
            user_vocab, user_vocab_stoi: user vocabulary and string-to-index mapping
            movie_title_dict: mapping from movie_id to title
            ratings_data: DataFrame with user_id, movie_ids (list), timestamps (list)
    """
    if not os.path.exists("ml-1m"):
        urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip")
        ZipFile("movielens.zip", "r").extractall()

    users = pd.read_csv(
        "ml-1m/users.dat", sep="::",
        names=["user_id", "sex", "age_group", "occupation", "zip_code"],
        engine="python",
    )
    ratings = pd.read_csv(
        "ml-1m/ratings.dat", sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
        engine="python",
    )
    movies = pd.read_csv(
        "ml-1m/movies.dat", sep="::",
        names=["movie_id", "title", "genres"],
        encoding="latin-1", engine="python",
    )

    # Prefix IDs so they are treated as strings, not numbers
    users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
    movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")
    ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
    ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")

    # Build vocabularies
    movie_vocab = SimpleVocab(Counter(movies.movie_id.unique()), specials=["<unk>"])
    movie_vocab_stoi = movie_vocab.get_stoi()

    user_vocab = SimpleVocab(Counter(users.user_id.unique()), specials=["<unk>"])
    user_vocab_stoi = user_vocab.get_stoi()

    movie_title_dict = dict(zip(movies.movie_id, movies.title))

    # Group ratings by user in chronological order
    ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")
    ratings_data = pd.DataFrame(data={
        "user_id": list(ratings_group.groups.keys()),
        "movie_ids": list(ratings_group.movie_id.apply(list)),
        "timestamps": list(ratings_group.unix_timestamp.apply(list)),
    })

    return {
        "users": users,
        "ratings": ratings,
        "movies": movies,
        "movie_vocab": movie_vocab,
        "movie_vocab_stoi": movie_vocab_stoi,
        "user_vocab": user_vocab,
        "user_vocab_stoi": user_vocab_stoi,
        "movie_title_dict": movie_title_dict,
        "ratings_data": ratings_data,
    }


# ============================================================
# Train / Validation / Test Splits
# ============================================================

def prepare_splits(ratings_data, seq_length, step_size=2, min_history=1):
    """Create sequences with a sliding window and split into train/val/test.

    Args:
        ratings_data: DataFrame with user_id and movie_ids columns.
        seq_length: Window size for sequence creation.
        step_size: Step size for the sliding window.
        min_history: Minimum history length to create a sequence.

    Returns:
        train_data, val_data, test_data: numpy arrays of (user_id, sequence) pairs.
    """
    rd = ratings_data.copy()
    rd["movie_ids"] = rd["movie_ids"].apply(
        lambda ids: create_sequences(ids, seq_length, step_size, min_history)
    )
    if "timestamps" in rd.columns:
        del rd["timestamps"]
    rd = rd[["user_id", "movie_ids"]].explode("movie_ids", ignore_index=True)
    rd.rename(columns={"movie_ids": "sequence_movie_ids"}, inplace=True)

    data = rd[["user_id", "sequence_movie_ids"]].values
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data
