import time
import math
import os
from datetime import datetime
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from collections import OrderedDict, Counter
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix



from zipfile import ZipFile
from urllib.request import urlretrieve

import pandas as pd
import numpy as np


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

# Downloading dataset (skip if already extracted)
if not os.path.exists("ml-1m"):
    urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip")
    ZipFile("movielens.zip", "r").extractall()

# Loading dataset
users = pd.read_csv(
    "ml-1m/users.dat",
    sep="::",
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
    engine="python",
)

ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    names=["user_id", "movie_id", "rating", "unix_timestamp"],
    engine="python",
)
movies = pd.read_csv(
    "ml-1m/movies.dat", sep="::", names=["movie_id", "title", "genres"],
    encoding='latin-1', engine="python",
)


# Preventing ids to be written as integer or float data type; We want the ids to be strings
users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")


movie_ids = movies.movie_id.unique() # Genarting a list of unique movie ids
movie_counter = Counter(movie_ids) # Counter is used to feed movies to movive_vocab
movie_vocab = SimpleVocab(movie_counter, specials=['<unk>']) # Generating vocabulary - maps each unique movie ID to a numerical index.
movie_vocab_stoi = movie_vocab.get_stoi() # string-to-index mapping for movies

movie_title_dict = dict(zip(movies.movie_id, movies.title)) # Movie to title mapping dictionary

# Similiary generating ids for user_ids
user_ids = users.user_id.unique()
user_counter = Counter(user_ids)
user_vocab = SimpleVocab(user_counter, specials=['<unk>'])
user_vocab_stoi = user_vocab.get_stoi()

# rating_ids
# Group ratings by user_id in order of increasing unix_timestamp.
ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id") # each group corresponds to a single user's ratings

ratings_data = pd.DataFrame(
    data={
        "user_id": list(ratings_group.groups.keys()),
        "movie_ids": list(ratings_group.movie_id.apply(list)), # all movies by a single user that is all movies in a rating group are added to a lisrt and that list is the value of the cell
        "timestamps": list(ratings_group.unix_timestamp.apply(list)), # same as movie ids. 
    }
)

# Sequence length, min history count and window slide size
sequence_length = 4
min_history = 1
step_size = 2

# Creating sequences from lists with sliding window
def create_sequences(values, window_size, step_size, min_history):
  sequences = []
  start_index = 0
  while len(values[start_index:]) > min_history:
    seq = values[start_index : start_index + window_size]
    sequences.append(seq)
    start_index += step_size
  return sequences

ratings_data.movie_ids = ratings_data.movie_ids.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size, min_history)
)

del ratings_data["timestamps"]

# Sub-sequences are exploded.
# Since there might be more than one sequence for each user.
ratings_data_transformed = ratings_data[["user_id", "movie_ids"]].explode(
    "movie_ids", ignore_index=True
)

ratings_data_transformed.rename(
    columns={"movie_ids": "sequence_movie_ids"},
    inplace=True,
)

# Start with the full dataset
data = ratings_data_transformed[["user_id", "sequence_movie_ids"]].values

# Step 1: Split into 60% train and 40% (temp)
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)

# Step 2: Split the remaining 40% into 20% validation and 20% test
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Print the sizes to confirm the split
print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")


# Pytorch Dataset for user interactions
class MovieSeqDataset(Dataset):
    # Initialize dataset
    def __init__(self, data, movie_vocab_stoi, user_vocab_stoi):
        self.data = data
        self.movie_vocab_stoi = movie_vocab_stoi
        self.user_vocab_stoi = user_vocab_stoi


    def __len__(self):
        return len(self.data)

    # Fetch data from the dataset
    def __getitem__(self, idx):
        user, movie_sequence = self.data[idx]
        # Directly index into the vocabularies
        movie_data = [self.movie_vocab_stoi[item] for item in movie_sequence]
        user_data = self.user_vocab_stoi[user]
        return torch.tensor(movie_data), torch.tensor(user_data)


# Collate function and padding
def collate_batch(batch):
    movie_list = [item[0] for item in batch]
    user_list = [item[1] for item in batch]
    return pad_sequence(movie_list, padding_value=movie_vocab_stoi['<unk>'], batch_first=True), torch.stack(user_list)


BATCH_SIZE = 256
# Create instances of your Dataset for each set
train_dataset = MovieSeqDataset(train_data, movie_vocab_stoi, user_vocab_stoi)
val_dataset = MovieSeqDataset(val_data, movie_vocab_stoi, user_vocab_stoi)
test_dataset = MovieSeqDataset(test_data, movie_vocab_stoi, user_vocab_stoi)

# Create DataLoaders
train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=collate_batch)
val_iter = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                      shuffle=False, collate_fn=collate_batch)
test_iter = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)

        # `div_term` distributes frequencies across embedding dimensions.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Positional encoding matrix: [1, max_len, d_model] for batch_first format.
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, nuser: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        # positional encoder
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Multihead attention with batch_first=True for correct [batch, seq, dim] handling.
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Embedding layers
        self.movie_embedding = nn.Embedding(ntoken, d_model)
        self.user_embedding = nn.Embedding(nuser, d_model)

        # Defining the size of the input to the model.
        self.d_model = d_model

        # Linear layer to map the output to movie vocabulary.
        self.linear = nn.Linear(2*d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        # Initializing the weights of the embedding and linear layers.
        initrange = 0.1
        self.movie_embedding.weight.data.uniform_(-initrange, initrange)
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generate an upper-triangular causal mask to prevent attending to future positions.

        This ensures that predictions at position i can only depend on positions 0..i,
        which is critical for autoregressive next-item prediction. Without this mask,
        the model can "cheat" by looking at future movies in the sequence.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src: Tensor, user: Tensor, src_mask: Tensor = None) -> Tensor:
        seq_len = src.size(1)  # src shape: [batch, seq_len]

        # Generate causal mask if not provided
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)

        # Embedding movie ids and userid
        movie_embed = self.movie_embedding(src) * math.sqrt(self.d_model)
        user_embed = self.user_embedding(user) * math.sqrt(self.d_model)

        # positional encoding
        movie_embed = self.pos_encoder(movie_embed)

        # Transformer encoder with causal mask to prevent future information leakage
        output = self.transformer_encoder(movie_embed, src_mask)

        # Expand user_embed tensor along the sequence length dimension
        user_embed = user_embed.expand(-1, output.size(1), -1)

        # Concatenate user embeddings with transformer output
        output = torch.cat((output, user_embed), dim=-1)

        output = self.linear(output)
        return output
    
ntokens = len(movie_vocab)  # size of vocabulary
nusers = len(user_vocab)
emsize = 128  # embedding dimension
d_hid = 128  # dimension of the feedforward network model
nlayers = 2  # number of ``nn.TransformerEncoderLayer``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(ntokens, nusers, emsize, nhead, d_hid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 1.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model: nn.Module, train_iter, epoch) -> None:
    # Switch to training mode
    model.train()
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    for i, (movie_data, user_data) in enumerate(train_iter):
        # Load movie sequence and user id
        movie_data, user_data = movie_data.to(device), user_data.to(device)
        user_data = user_data.reshape(-1, 1)

        # Split movie sequence to inputs and targets
        inputs, targets = movie_data[:, :-1], movie_data[:, 1:]
        targets_flat = targets.reshape(-1)

        # Predict movies
        output = model(inputs, user_data)
        output_flat = output.reshape(-1, ntokens)
        
        # Backpropogation process
        loss = criterion(output_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        # Results
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    # Switch the model to evaluation mode.
    # This is necessary for layers like dropout,
    model.eval() 
    total_loss = 0.

    with torch.no_grad():
        for i, (movie_data, user_data) in enumerate(eval_data):
            # Load movie sequence and user id
            movie_data, user_data = movie_data.to(device), user_data.to(device)
            user_data = user_data.reshape(-1, 1)
            # Split movie sequence to inputs and targets
            inputs, targets = movie_data[:, :-1], movie_data[:, 1:]
            targets_flat = targets.reshape(-1)
            # Predict movies
            output = model(inputs, user_data)
            output_flat = output.reshape(-1, ntokens)
            # Calculate loss
            loss = criterion(output_flat, targets_flat)
            total_loss += loss.item()
    return total_loss / (len(eval_data) - 1)


best_val_loss = float('inf')
epochs = 10

# Save checkpoints and results to project directory for later comparison
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
best_model_params_path = os.path.join(CHECKPOINT_DIR, "best_model_params.pt")

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    # Training
    train(model, train_iter, epoch)

    # Evaluation
    val_loss = evaluate(model, val_iter)
    test_loss = evaluate(model, test_iter)

    # Compute the perplexity of the validation loss
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time

    # Results
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_params_path)

    scheduler.step()

# Combine train and validation data for final training
final_train_data = np.concatenate([train_data, val_data], axis=0)
final_train_dataset = MovieSeqDataset(final_train_data, movie_vocab_stoi, user_vocab_stoi)
final_train_iter = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

print("\nStarting final training on combined train+validation set...")
final_epochs = 5

for epoch in range(1, final_epochs + 1):
    epoch_start_time = time.time()
    train(model, final_train_iter, epoch)
    elapsed = time.time() - epoch_start_time
    print(f"| Final Training Epoch {epoch} completed in {elapsed:.2f}s |")

# Save final model parameters to project directory (persisted for later use)
final_model_params_path = os.path.join(CHECKPOINT_DIR, "final_model_params.pt")
torch.save(model.state_dict(), final_model_params_path)
print(f"\nSaved final model to {final_model_params_path}")

# Load final model parameters
model.load_state_dict(torch.load(final_model_params_path))

# Evaluate on the test set
print("\nEvaluating the final model on the test set...")
final_test_loss = evaluate(model, test_iter)
print(f"Final Test Loss (after train+val training): {final_test_loss:.4f}")
print(f"Final Test Perplexity (after train+val training): {math.exp(final_test_loss):.2f}")

# ============================================================
# Baseline Models
# ============================================================

# --- Baseline 1: Popularity ---
def get_popular_movies(df_ratings):
    """Get top-10 movies by average rating among frequently rated movies (95th percentile)."""
    rating_counts = df_ratings['movie_id'].value_counts().reset_index()
    rating_counts.columns = ['movie_id', 'rating_count']
    min_ratings_threshold = rating_counts['rating_count'].quantile(0.95)
    popular_movies = ratings.merge(rating_counts, on='movie_id')
    popular_movies = popular_movies[popular_movies['rating_count'] >= min_ratings_threshold]
    average_ratings = popular_movies.groupby('movie_id')['rating'].mean().reset_index()
    top_10_movies = list(average_ratings.sort_values('rating', ascending=False).head(10).movie_id.values)
    return top_10_movies

top_10_movies = get_popular_movies(ratings)

# --- Baseline 2: First-Order Markov Chain ---
def build_markov_chain(train_data, movie_vocab_stoi):
    """Build first-order Markov transition counts from training sequences.

    For each consecutive pair (A -> B) in a user's watch history, records that
    movie A was followed by movie B. Works with vocabulary indices for efficiency.
    At prediction time, given the last watched movie, recommends movies with the
    highest transition counts.
    """
    transitions = {}
    for user, sequence in train_data:
        indices = [movie_vocab_stoi[m] for m in sequence]
        for i in range(len(indices) - 1):
            curr = indices[i]
            nxt = indices[i + 1]
            if curr not in transitions:
                transitions[curr] = Counter()
            transitions[curr][nxt] += 1
    return transitions

print("Building Markov chain baseline...")
markov_transitions = build_markov_chain(train_data, movie_vocab_stoi)
print(f"  Markov chain covers {len(markov_transitions)} unique source movies.")

# --- Baseline 3: SVD Collaborative Filtering ---
def build_svd_model(train_data, movie_vocab_stoi, user_vocab_stoi, n_components=50):
    """Build a truncated SVD model on the binary user-movie interaction matrix.

    Uses only training-set interactions to avoid data leakage. The interaction
    matrix is binary (1 if user watched movie in training, 0 otherwise).
    SVD factorizes this into user and item latent factors that capture
    collaborative filtering signals.
    """
    n_users = len(user_vocab_stoi)
    n_movies = len(movie_vocab_stoi)

    rows, cols = [], []
    for user, sequence in train_data:
        user_idx = user_vocab_stoi[user]
        for movie in sequence:
            movie_idx = movie_vocab_stoi[movie]
            rows.append(user_idx)
            cols.append(movie_idx)

    vals = np.ones(len(rows), dtype=np.float32)
    interaction_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(interaction_matrix)   # [n_users, n_components]
    item_factors = svd.components_                          # [n_components, n_movies]

    explained_var = svd.explained_variance_ratio_.sum()
    print(f"  SVD explained variance ratio: {explained_var:.4f}")

    return user_factors, item_factors

print("Building SVD collaborative filtering baseline...")
user_factors, item_factors = build_svd_model(train_data, movie_vocab_stoi, user_vocab_stoi)

# ============================================================
# Evaluation: Compare All Models
# ============================================================
print("\n" + "=" * 89)
print("Evaluating all models on validation set...")
print("=" * 89)

movie_vocab_itos = movie_vocab.get_itos()

# Results containers for each model
reco_results = {
    'Transformer': [],
    'Popularity': [],
    'Markov Chain': [],
    'SVD (CF)': [],
}

k_max = 10  # Maximum K for evaluation

model.eval()
n_val_batches = len(val_iter)
with torch.no_grad():
    for i, (movie_data, user_data) in enumerate(val_iter):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Evaluation batch {i + 1}/{n_val_batches}...")
        movie_data, user_data = movie_data.to(device), user_data.to(device)
        user_indices = user_data.clone()   # [batch] — preserve for SVD baseline
        user_data = user_data.reshape(-1, 1)

        inputs, targets = movie_data[:, :-1], movie_data[:, 1:]

        # --- Transformer predictions ---
        output = model(inputs, user_data)
        output_flat = output.reshape(-1, ntokens)

        # Get predictions at the last sequence position
        outputs_last = output_flat.reshape(
            output_flat.shape[0] // inputs.shape[1],
            inputs.shape[1],
            output_flat.shape[1]
        )[:, -1, :]

        _, transformer_top_indices = outputs_last.topk(k_max + inputs.shape[1], dim=-1)

        # --- Per-sequence evaluation across all models ---
        for sub_sequence, t_indices, user_idx in zip(
            movie_data.cpu().numpy(),
            transformer_top_indices.cpu().numpy(),
            user_indices.cpu().numpy()
        ):
            input_movies = sub_sequence[:-1]
            target_movie = sub_sequence[-1]

            # 1. Transformer: filter already-watched, take top-k
            mask = np.isin(t_indices, input_movies, invert=True)
            transformer_topk = t_indices[mask][:k_max]
            transformer_hit = np.isin(transformer_topk, target_movie).astype(int)
            if len(transformer_hit) < k_max:
                transformer_hit = np.pad(transformer_hit, (0, k_max - len(transformer_hit)))
            reco_results['Transformer'].append(transformer_hit)

            # 2. Popularity: static top-10 recommendation
            target_decoded = movie_vocab_itos[target_movie]
            popular_hit = np.isin(top_10_movies, target_decoded).astype(int)
            if len(popular_hit) < k_max:
                popular_hit = np.pad(popular_hit, (0, k_max - len(popular_hit)))
            reco_results['Popularity'].append(popular_hit)

            # 3. Markov Chain: predict based on last input movie's transitions
            last_input = input_movies[-1]
            if last_input in markov_transitions:
                markov_preds = [
                    m for m, _ in markov_transitions[last_input].most_common()
                    if m not in input_movies
                ][:k_max]
            else:
                markov_preds = []
            markov_hit = np.isin(markov_preds, target_movie).astype(int)
            if len(markov_hit) < k_max:
                markov_hit = np.pad(markov_hit, (0, k_max - len(markov_hit)))
            reco_results['Markov Chain'].append(markov_hit)

            # 4. SVD Collaborative Filtering: predict based on user latent factors
            svd_scores = user_factors[user_idx] @ item_factors   # [n_movies]
            svd_scores[input_movies] = -np.inf                   # exclude watched
            svd_topk = np.argsort(svd_scores)[::-1][:k_max]
            svd_hit = np.isin(svd_topk, target_movie).astype(int)
            reco_results['SVD (CF)'].append(svd_hit)

# ============================================================
# Metrics: NDCG@K, Precision@K, Hit Rate@K for all models
# ============================================================
print("\n" + "=" * 89)
print(f"{'Model':<16} {'Metric':<14} {'@3':>8} {'@5':>8} {'@10':>8}")
print("=" * 89)

representative_array = [[i for i in range(k_max, 0, -1)]] * len(reco_results['Transformer'])

# Collect metrics for CSV (one row per model) so you can compare runs
results_rows = []
run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for model_name in ['Transformer', 'Markov Chain', 'SVD (CF)', 'Popularity']:
    results_arr = reco_results[model_name]

    ndcg_vals, prec_vals, hr_vals = {}, {}, {}

    for k in [3, 5, 10]:
        # NDCG@K
        ndcg_vals[k] = ndcg_score(results_arr, representative_array, k=k)

        # Precision@K and Hit Rate@K
        precisions, hits = [], []
        for result in results_arr:
            precisions.append(np.sum(result[:k]) / k)
            hits.append(1.0 if np.sum(result[:k]) > 0 else 0.0)
        prec_vals[k] = np.mean(precisions)
        hr_vals[k] = np.mean(hits)

    print(f"{model_name:<16} {'NDCG':<14} {ndcg_vals[3]:>8.4f} {ndcg_vals[5]:>8.4f} {ndcg_vals[10]:>8.4f}")
    print(f"{'':<16} {'Precision':<14} {prec_vals[3]:>8.4f} {prec_vals[5]:>8.4f} {prec_vals[10]:>8.4f}")
    print(f"{'':<16} {'Hit Rate':<14} {hr_vals[3]:>8.4f} {hr_vals[5]:>8.4f} {hr_vals[10]:>8.4f}")
    print("-" * 89)

    results_rows.append({
        "run": run_ts,
        "model": model_name,
        "NDCG@3": ndcg_vals[3], "NDCG@5": ndcg_vals[5], "NDCG@10": ndcg_vals[10],
        "Precision@3": prec_vals[3], "Precision@5": prec_vals[5], "Precision@10": prec_vals[10],
        "HitRate@3": hr_vals[3], "HitRate@5": hr_vals[5], "HitRate@10": hr_vals[10],
    })

# Save results to CSV for easy comparison across runs
results_df = pd.DataFrame(results_rows)
results_path = os.path.join(RESULTS_DIR, f"recommendation_results_{run_ts}.csv")
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to {results_path}")
print("To compare runs: load all recommendation_results_*.csv files and compare by 'model' and 'run'.")