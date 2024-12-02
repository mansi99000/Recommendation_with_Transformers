import time
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.vocab import vocab
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split



from collections import Counter

from zipfile import ZipFile
from urllib.request import urlretrieve

import pandas as pd
import numpy as np

# Downloading dataset
urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip")
ZipFile("movielens.zip", "r").extractall()

# Loading dataset
users = pd.read_csv(
    "ml-1m/users.dat",
    sep="::",
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
)

ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    names=["user_id", "movie_id", "rating", "unix_timestamp"],
)
movies = pd.read_csv(
    "ml-1m/movies.dat", sep="::", names=["movie_id", "title", "genres"], encoding='latin-1'
)


# Preventing ids to be written as integer or float data type; We want the ids to be strings
users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")


movie_ids = movies.movie_id.unique() # Genarting a list of unique movie ids
movie_counter = Counter(movie_ids) # Counter is used to feed movies to movive_vocab
movie_vocab = vocab(movie_counter, specials=['<unk>']) # Genarting vocabulary - The vocab object maps each unique movie ID to a numerical index.
movie_vocab_stoi = movie_vocab.get_stoi() # string-to-index mapping for movies

movie_title_dict = dict(zip(movies.movie_id, movies.title)) # Movie to title mapping dictionary

# Similiary generating ids for user_ids
user_ids = users.user_id.unique()
user_counter = Counter(user_ids)
user_vocab = vocab(user_counter, specials=['<unk>'])
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

        # `div_term` is used in the calculation of the sinusoidal values.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Initializing positional encoding matrix with zeros.
        pe = torch.zeros(max_len, 1, d_model)

        # Calculating the positional encodings.
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, nuser: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        # positional encoder
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Multihead attention mechanism.
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Embedding layers
        self.movie_embedding = nn.Embedding(ntoken, d_model)
        self.user_embedding = nn.Embedding(nuser, d_model)

        # Defining the size of the input to the model.
        self.d_model = d_model

        # Linear layer to map the output tomovie vocabulary.
        self.linear = nn.Linear(2*d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        # Initializing the weights of the embedding and linear layers.
        initrange = 0.1
        self.movie_embedding.weight.data.uniform_(-initrange, initrange)
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, user: Tensor, src_mask: Tensor = None) -> Tensor:
        # Embedding movie ids and userid
        movie_embed = self.movie_embedding(src) * math.sqrt(self.d_model)
        user_embed = self.user_embedding(user) * math.sqrt(self.d_model)

        # positional encoding
        movie_embed = self.pos_encoder(movie_embed)

        # generating output with final layers
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

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

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
    # model.load_state_dict(torch.load(best_model_params_path)) # load best model states
    # test_loss = evaluate(model, test_iter)
    # print(f'Final Test loss: {test_loss:.4f}')
    # print(f'Final Test perplexity: {math.exp(test_loss):.2f}')

    # Combine train and validation data for final training
    final_train_data = np.concatenate([train_data, val_data], axis=0)  # Combine train and validation data
    final_train_dataset = MovieSeqDataset(final_train_data, movie_vocab_stoi, user_vocab_stoi)
    final_train_iter = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    print("\nStarting final training on combined train+validation set...")
    final_epochs = 5  # Set the number of additional epochs for final training

    for epoch in range(1, final_epochs + 1):
        epoch_start_time = time.time()
        train(model, final_train_iter, epoch)
        elapsed = time.time() - epoch_start_time
        print(f"| Final Training Epoch {epoch} completed in {elapsed:.2f}s |")

    # Save final model parameters after training on train+validation
    final_model_params_path = os.path.join(tempdir, "final_model_params.pt")
    torch.save(model.state_dict(), final_model_params_path)

    # Load final model parameters
    model.load_state_dict(torch.load(final_model_params_path))

    # Evaluate on the test set
    print("\nEvaluating the final model on the test set...")
    final_test_loss = evaluate(model, test_iter)
    print(f"Final Test Loss (after train+val training): {final_test_loss:.4f}")
    print(f"Final Test Perplexity (after train+val training): {math.exp(final_test_loss):.2f}")

def get_popular_movies(df_ratings):
  # Calculate the number of ratings for each movie
  rating_counts = df_ratings['movie_id'].value_counts().reset_index()
  rating_counts.columns = ['movie_id', 'rating_count']

  # Get the most frequently rated movies
  min_ratings_threshold = rating_counts['rating_count'].quantile(0.95)

  # Filter movies based on the minimum number of ratings
  popular_movies = ratings.merge(rating_counts, on='movie_id')
  popular_movies = popular_movies[popular_movies['rating_count'] >= min_ratings_threshold]

  # Calculate the average rating for each movie
  average_ratings = popular_movies.groupby('movie_id')['rating'].mean().reset_index()

  # Get the top 10 rated movies
  top_10_movies = list(average_ratings.sort_values('rating', ascending=False).head(10).movie_id.values)
  return top_10_movies

top_10_movies = get_popular_movies(ratings)

# Movie id decoder
movie_vocab_itos = movie_vocab.get_itos()

# A placeholders to store results of recommendations
transformer_reco_results = list()
popular_reco_results = list()

# Get top 10 movies
k = 10
# Iterate over the validation data
for i, (movie_data, user_data) in enumerate(val_iter): 
    # Feed the input and get the outputs
    movie_data, user_data = movie_data.to(device), user_data.to(device)
    user_data = user_data.reshape(-1, 1)
    inputs, targets = movie_data[:, :-1], movie_data[:, 1:]
    output = model(inputs, user_data)
    output_flat = output.reshape(-1, ntokens)
    targets_flat = targets.reshape(-1)

    # Reshape the output_flat to get top predictions
    outputs = output_flat.reshape(output_flat.shape[0] // inputs.shape[1],
                                  inputs.shape[1],
                                  output_flat.shape[1])[: , -1, :]
    # k + inputs.shape[1] = 13 movies obtained
    # In order to prevent to recommend already watched movies
    values, indices = outputs.topk(k + inputs.shape[1], dim=-1)

    for sub_sequence, sub_indice_org in zip(movie_data, indices):
        sub_indice_org = sub_indice_org.cpu().detach().numpy()
        sub_sequence = sub_sequence.cpu().detach().numpy()

        # Generate mask array to eliminate already watched movies 
        mask = np.isin(sub_indice_org, sub_sequence[:-1], invert=True)

        # After masking get top k movies
        sub_indice = sub_indice_org[mask][:k]

        # Generate results array
        transformer_reco_result = np.isin(sub_indice, sub_sequence[-1]).astype(int)

        # Decode movie to search in popular movies
        target_movie_decoded = movie_vocab_itos[sub_sequence[-1]]
        popular_reco_result = np.isin(top_10_movies, target_movie_decoded).astype(int)

        transformer_reco_results.append(transformer_reco_result)
        popular_reco_results.append(popular_reco_result)


# Since we have already sorted our recommendations
# An array that represent our recommendation scores is used.
representative_array = [[i for i in range(k, 0, -1)]] * len(transformer_reco_results)

# Placeholder for precision results
transformer_precision_results = []
popular_precision_results = []

# for k in [3, 5, 10]:
#     transformer_result = ndcg_score(transformer_reco_results,
#                                     representative_array, k=k)
#     popular_result = ndcg_score(popular_reco_results,
#                                 representative_array, k=k)
    

  
# print(f"Transformer NDCG result at top {k}: {round(transformer_result, 4)}")
# print(f"Popular recommendation NDCG result at top {k}: {round(popular_result, 4)}\n\n")

for k in [3, 5, 10]:
    # Calculate NDCG
    transformer_result = ndcg_score(transformer_reco_results,
                                    representative_array, k=k)
    popular_result = ndcg_score(popular_reco_results,
                                representative_array, k=k)

    # Calculate Precision@K for Transformer and Popular recommendations
    transformer_precision = []
    popular_precision = []

    for transformer_reco, popular_reco in zip(transformer_reco_results, popular_reco_results):
        # Calculate Precision for Transformer
        transformer_precision_at_k = np.sum(transformer_reco[:k]) / k
        transformer_precision.append(transformer_precision_at_k)

        # Calculate Precision for Popular
        popular_precision_at_k = np.sum(popular_reco[:k]) / k
        popular_precision.append(popular_precision_at_k)

    # Average Precision@K across all samples
    transformer_precision_avg = np.mean(transformer_precision)
    popular_precision_avg = np.mean(popular_precision)

    print(f"Transformer NDCG result at top {k}: {round(transformer_result, 4)}")
    print(f"Popular recommendation NDCG result at top {k}: {round(popular_result, 4)}")
    print(f"Transformer Precision@{k}: {round(transformer_precision_avg, 4)}")
    print(f"Popular Precision@{k}: {round(popular_precision_avg, 4)}\n\n")