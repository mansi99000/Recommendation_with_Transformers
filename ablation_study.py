"""
Ablation Study: Impact of Hyperparameters on Transformer-Based Recommendations
===============================================================================

Systematically varies one hyperparameter at a time while keeping others at
default values. Reports validation loss, perplexity, and Hit Rate@10 for
each configuration.

Default config:  emsize=128, nlayers=2, nhead=2, seq_len=4, dropout=0.2
Ablation groups:
  - Embedding dimension: [64, 128, 256]
  - Number of layers:    [1, 2, 4]
  - Number of heads:     [1, 2, 4]
  - Sequence length:     [4, 8, 16]

Usage:
    python3 ablation_study.py
"""

import time
import math
import os
from collections import Counter
from datetime import datetime

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from urllib.request import urlretrieve

import pandas as pd
import numpy as np


# ============================================================
# Shared Components (same as recommendation.py)
# ============================================================

class SimpleVocab:
    """Lightweight replacement for torchtext.vocab.vocab (deprecated)."""
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, nuser, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.movie_embedding = nn.Embedding(ntoken, d_model)
        self.user_embedding = nn.Embedding(nuser, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(2 * d_model, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.movie_embedding.weight.data.uniform_(-initrange, initrange)
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src, user, src_mask=None):
        seq_len = src.size(1)
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        movie_embed = self.movie_embedding(src) * math.sqrt(self.d_model)
        user_embed = self.user_embedding(user) * math.sqrt(self.d_model)
        movie_embed = self.pos_encoder(movie_embed)
        output = self.transformer_encoder(movie_embed, src_mask)
        user_embed = user_embed.expand(-1, output.size(1), -1)
        output = torch.cat((output, user_embed), dim=-1)
        output = self.linear(output)
        return output


class MovieSeqDataset(Dataset):
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
# Data Loading
# ============================================================

def load_data():
    """Load and preprocess MovieLens 1M dataset."""
    if not os.path.exists("ml-1m"):
        urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip")
        ZipFile("movielens.zip", "r").extractall()

    users = pd.read_csv("ml-1m/users.dat", sep="::",
                         names=["user_id", "sex", "age_group", "occupation", "zip_code"],
                         engine="python")
    ratings = pd.read_csv("ml-1m/ratings.dat", sep="::",
                           names=["user_id", "movie_id", "rating", "unix_timestamp"],
                           engine="python")
    movies = pd.read_csv("ml-1m/movies.dat", sep="::",
                          names=["movie_id", "title", "genres"],
                          encoding='latin-1', engine="python")

    users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
    movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")
    ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
    ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")

    movie_ids = movies.movie_id.unique()
    movie_counter = Counter(movie_ids)
    movie_vocab = SimpleVocab(movie_counter, specials=['<unk>'])
    movie_vocab_stoi = movie_vocab.get_stoi()

    user_ids = users.user_id.unique()
    user_counter = Counter(user_ids)
    user_vocab = SimpleVocab(user_counter, specials=['<unk>'])
    user_vocab_stoi = user_vocab.get_stoi()

    ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")
    ratings_data = pd.DataFrame(data={
        "user_id": list(ratings_group.groups.keys()),
        "movie_ids": list(ratings_group.movie_id.apply(list)),
        "timestamps": list(ratings_group.unix_timestamp.apply(list)),
    })

    return ratings_data, movie_vocab, movie_vocab_stoi, user_vocab, user_vocab_stoi


def create_sequences(values, window_size, step_size, min_history=1):
    """Create sliding-window sequences from a list of movie IDs."""
    sequences = []
    start_index = 0
    while len(values[start_index:]) > min_history:
        seq = values[start_index : start_index + window_size]
        sequences.append(seq)
        start_index += step_size
    return sequences


def prepare_splits(ratings_data, seq_length, step_size=2):
    """Create sequences and split into train/val/test."""
    rd = ratings_data.copy()
    rd.movie_ids = rd.movie_ids.apply(
        lambda ids: create_sequences(ids, seq_length, step_size, min_history=1)
    )
    if "timestamps" in rd.columns:
        del rd["timestamps"]
    rd = rd[["user_id", "movie_ids"]].explode("movie_ids", ignore_index=True)
    rd.rename(columns={"movie_ids": "sequence_movie_ids"}, inplace=True)

    data = rd[["user_id", "sequence_movie_ids"]].values
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data


# ============================================================
# Training & Evaluation
# ============================================================

def train_one_epoch(model, train_iter, criterion, optimizer, ntokens, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.
    n_batches = 0
    for movie_data, user_data in train_iter:
        movie_data, user_data = movie_data.to(device), user_data.to(device)
        user_data = user_data.reshape(-1, 1)
        inputs, targets = movie_data[:, :-1], movie_data[:, 1:]
        targets_flat = targets.reshape(-1)

        output = model(inputs, user_data)
        output_flat = output.reshape(-1, ntokens)

        loss = criterion(output_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_loss(model, eval_iter, criterion, ntokens, device):
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.
    n_batches = 0
    with torch.no_grad():
        for movie_data, user_data in eval_iter:
            movie_data, user_data = movie_data.to(device), user_data.to(device)
            user_data = user_data.reshape(-1, 1)
            inputs, targets = movie_data[:, :-1], movie_data[:, 1:]
            targets_flat = targets.reshape(-1)

            output = model(inputs, user_data)
            output_flat = output.reshape(-1, ntokens)
            loss = criterion(output_flat, targets_flat)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_hit_rate(model, eval_iter, ntokens, device, k=10):
    """Compute Hit Rate@K on evaluation data."""
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for movie_data, user_data in eval_iter:
            movie_data, user_data = movie_data.to(device), user_data.to(device)
            user_data_r = user_data.reshape(-1, 1)
            inputs, targets = movie_data[:, :-1], movie_data[:, 1:]

            output = model(inputs, user_data_r)
            output_flat = output.reshape(-1, ntokens)
            outputs_last = output_flat.reshape(
                output_flat.shape[0] // inputs.shape[1],
                inputs.shape[1],
                output_flat.shape[1]
            )[:, -1, :]

            _, top_indices = outputs_last.topk(k + inputs.shape[1], dim=-1)

            for sub_seq, t_idx in zip(movie_data.cpu().numpy(), top_indices.cpu().numpy()):
                input_movies = sub_seq[:-1]
                target = sub_seq[-1]
                mask = np.isin(t_idx, input_movies, invert=True)
                topk = t_idx[mask][:k]
                if target in topk:
                    hits += 1
                total += 1
    return hits / max(total, 1)


# ============================================================
# Run One Ablation Configuration
# ============================================================

def run_config(config, train_data, val_data, movie_vocab_stoi, user_vocab_stoi,
               ntokens, nusers, device, ablation_epochs=3, batch_size=256):
    """Train a model with the given config and return metrics."""
    unk_idx = movie_vocab_stoi['<unk>']

    def collate_fn(batch):
        movies = [item[0] for item in batch]
        users = [item[1] for item in batch]
        return pad_sequence(movies, padding_value=unk_idx, batch_first=True), torch.stack(users)

    train_dataset = MovieSeqDataset(train_data, movie_vocab_stoi, user_vocab_stoi)
    val_dataset = MovieSeqDataset(val_data, movie_vocab_stoi, user_vocab_stoi)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerModel(
        ntoken=ntokens, nuser=nusers,
        d_model=config['emsize'], nhead=config['nhead'],
        d_hid=config['emsize'], nlayers=config['nlayers'],
        dropout=config['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    start = time.time()
    for epoch in range(1, ablation_epochs + 1):
        train_one_epoch(model, train_iter, criterion, optimizer, ntokens, device)
        scheduler.step()
    elapsed = time.time() - start

    val_loss = evaluate_loss(model, val_iter, criterion, ntokens, device)
    val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
    hr10 = evaluate_hit_rate(model, val_iter, ntokens, device, k=10)

    return {
        'val_loss': val_loss,
        'val_ppl': val_ppl,
        'hr@10': hr10,
        'time_s': elapsed,
    }


# ============================================================
# Main: Ablation Experiment Grid
# ============================================================

if __name__ == "__main__":
    print("=" * 89)
    print("Ablation Study: Transformer-Based Movie Recommendations")
    print("=" * 89)

    # --- Load data ---
    print("\nLoading MovieLens 1M dataset...")
    ratings_data_raw, movie_vocab, movie_vocab_stoi, user_vocab, user_vocab_stoi = load_data()
    ntokens = len(movie_vocab)
    nusers = len(user_vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Movies: {ntokens}, Users: {nusers}, Device: {device}")

    # --- Default hyperparameters ---
    DEFAULT = {
        'emsize': 128,
        'nlayers': 2,
        'nhead': 2,
        'seq_len': 4,
        'dropout': 0.2,
    }

    ABLATION_EPOCHS = 3   # Fewer epochs for speed; enough for relative comparison

    # --- Define ablation grid (vary one param at a time) ---
    experiments = []

    # Embedding dimension
    for emsize in [64, 128, 256]:
        cfg = {**DEFAULT, 'emsize': emsize}
        experiments.append(('emsize', emsize, cfg))

    # Number of layers
    for nlayers in [1, 2, 4]:
        cfg = {**DEFAULT, 'nlayers': nlayers}
        if cfg != {**DEFAULT}:  # skip duplicate of default
            experiments.append(('nlayers', nlayers, cfg))

    # Number of attention heads
    for nhead in [1, 2, 4]:
        cfg = {**DEFAULT, 'nhead': nhead}
        if nhead != DEFAULT['nhead']:
            experiments.append(('nhead', nhead, cfg))

    # Sequence length
    for seq_len in [4, 8, 16]:
        cfg = {**DEFAULT, 'seq_len': seq_len}
        if seq_len != DEFAULT['seq_len']:
            experiments.append(('seq_len', seq_len, cfg))

    # --- Run experiments ---
    results = []
    total = len(experiments)

    for idx, (param_name, param_val, config) in enumerate(experiments, 1):
        label = f"{param_name}={param_val}"
        print(f"\n[{idx}/{total}] Running config: {label}")
        print(f"  Config: emsize={config['emsize']}, nlayers={config['nlayers']}, "
              f"nhead={config['nhead']}, seq_len={config['seq_len']}, dropout={config['dropout']}")

        # Prepare data with the configured sequence length
        train_data, val_data, _ = prepare_splits(
            ratings_data_raw, seq_length=config['seq_len']
        )
        print(f"  Train sequences: {len(train_data)}, Val sequences: {len(val_data)}")

        metrics = run_config(
            config, train_data, val_data,
            movie_vocab_stoi, user_vocab_stoi,
            ntokens, nusers, device,
            ablation_epochs=ABLATION_EPOCHS,
        )

        results.append((label, param_name, param_val, config, metrics))
        print(f"  -> Val Loss: {metrics['val_loss']:.4f} | "
              f"PPL: {metrics['val_ppl']:.2f} | "
              f"HR@10: {metrics['hr@10']:.4f} | "
              f"Time: {metrics['time_s']:.1f}s")

    # --- Summary Table ---
    print("\n\n" + "=" * 89)
    print("ABLATION STUDY RESULTS")
    print("=" * 89)
    print(f"{'Config':<20} {'Val Loss':>10} {'PPL':>10} {'HR@10':>10} {'Time (s)':>10}")
    print("-" * 89)

    for label, param_name, param_val, config, metrics in results:
        tag = " *" if config == DEFAULT else ""
        print(f"{label + tag:<20} {metrics['val_loss']:>10.4f} {metrics['val_ppl']:>10.2f} "
              f"{metrics['hr@10']:>10.4f} {metrics['time_s']:>10.1f}")

    print("-" * 89)
    print("* = default configuration")
    print(f"\nAll experiments used {ABLATION_EPOCHS} training epochs.")

    # --- Per-parameter analysis ---
    print("\n\n" + "=" * 89)
    print("PER-PARAMETER ANALYSIS")
    print("=" * 89)

    param_groups = {}
    for label, param_name, param_val, config, metrics in results:
        if param_name not in param_groups:
            param_groups[param_name] = []
        param_groups[param_name].append((param_val, metrics))

    for param_name, entries in param_groups.items():
        print(f"\n--- {param_name} ---")
        best_val = min(entries, key=lambda x: x[1]['val_loss'])
        best_hr = max(entries, key=lambda x: x[1]['hr@10'])
        print(f"  Best validation loss:  {param_name}={best_val[0]} (loss={best_val[1]['val_loss']:.4f})")
        print(f"  Best Hit Rate@10:      {param_name}={best_hr[0]} (HR@10={best_hr[1]['hr@10']:.4f})")

    # --- Save results to CSV for easy comparison across runs ---
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ablation_path = os.path.join(RESULTS_DIR, f"ablation_results_{run_ts}.csv")

    ablation_rows = []
    for label, param_name, param_val, config, metrics in results:
        ablation_rows.append({
            "run": run_ts,
            "config": label,
            "param_name": param_name,
            "param_value": param_val,
            "val_loss": metrics["val_loss"],
            "val_ppl": metrics["val_ppl"],
            "hr_at_10": metrics["hr@10"],
            "time_s": metrics["time_s"],
        })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(ablation_path, index=False)

    print("\n" + "=" * 89)
    print("Ablation study complete.")
    print(f"Results saved to {ablation_path}")
    print("To compare runs: load all ablation_results_*.csv files and compare by 'config' or 'param_name'.")
