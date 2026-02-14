import math

import torch
from torch import nn, Tensor

# PositionalEncoding adds information about the *order* of movies in a sequence.
# Transformers have no built-in notion of position (unlike RNNs which process
# tokens one-by-one), so we inject sine/cosine signals to tell the model
# "this is the 1st movie watched, this is the 2nd," etc.
from positional_encoder import PositionalEncoding

# TransformerEncoderLayer: A single layer of the Transformer encoder. Each layer
#   contains two sub-layers:
#     1. Multi-Head Self-Attention -- lets every movie in the sequence "look at"
#        every other movie to learn relationships (e.g., genre patterns).
#     2. Position-wise Feed-Forward Network -- a two-layer MLP applied to each
#        position independently, adding non-linear capacity.
#   Each sub-layer has a residual connection + layer normalization around it.
#
# TransformerEncoder: Stacks multiple TransformerEncoderLayers on top of each
#   other. More layers = deeper representation, letting the model capture more
#   complex sequential patterns.
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    """
    Transformer-based sequential recommendation model.

    High-level idea:
      Given a user and their recent watch history (a sequence of movie IDs),
      predict a probability distribution over ALL movies for "what to watch next."

    Architecture overview:
      1. Movie IDs  -->  movie_embedding  -->  positional encoding
      2. Feed into a stack of Transformer encoder layers (self-attention + FFN)
      3. Concatenate the Transformer output with a learned user embedding
      4. Project through a linear layer to get scores for every movie

    Args:
        ntoken: Number of unique movies (size of the movie vocabulary).
                This determines the embedding table size and the output size.
        nuser:  Number of unique users. Each user gets their own learned vector.
        d_model: Dimension of embeddings and internal Transformer representations.
                 Larger = more expressive but slower and more memory.
        nhead:  Number of attention heads. Multi-head attention lets the model
                attend to different aspects of the sequence simultaneously.
                d_model must be divisible by nhead (each head gets d_model/nhead dims).
        d_hid:  Hidden dimension of the feed-forward network inside each
                Transformer layer. Typically equal to or larger than d_model.
        nlayers: How many Transformer encoder layers to stack. More layers let the
                 model capture longer-range and more abstract patterns.
        dropout: Probability of zeroing out activations during training (regularization).
                 Helps prevent overfitting. Not applied during inference.
    """

    def __init__(self, ntoken: int, nuser: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        # nn.Module.__init__() -- required by PyTorch; registers this as a proper
        # module so PyTorch can track all parameters, move them to GPU, etc.
        super().__init__()

        self.model_type = 'Transformer'

        # Positional encoding layer: injects position info into the embeddings.
        # Without this, the Transformer would treat the sequence as a "bag of movies"
        # and lose all ordering information.
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # --- Build the Transformer Encoder ---
        # Step 1: Define ONE encoder layer.
        #   - d_model:  input/output dimension for this layer
        #   - nhead:    number of parallel attention heads
        #   - d_hid:    hidden size of the feed-forward sub-layer
        #   - dropout:  applied inside attention and feed-forward
        #   - batch_first=True: tells PyTorch our tensors are shaped
        #     [batch_size, seq_len, d_model] rather than [seq_len, batch_size, d_model].
        #     This is just a convenience so we don't have to transpose everywhere.
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)

        # Step 2: Stack `nlayers` copies of that layer into a full encoder.
        # PyTorch clones the layer definition internally, so each layer gets its
        # own independent set of weights.
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # --- Embedding Layers ---
        # nn.Embedding is a lookup table: given an integer ID, return a dense vector.
        # Internally it's just a weight matrix of shape [num_items, d_model].
        # These vectors are *learned* during training -- the model discovers which
        # movies are similar by placing them near each other in this vector space.

        # Movie embedding: maps each movie ID (0..ntoken-1) to a d_model-dim vector.
        self.movie_embedding = nn.Embedding(ntoken, d_model)

        # User embedding: maps each user ID to a d_model-dim vector.
        # This captures user-specific preferences (e.g., "this user likes action movies").
        self.user_embedding = nn.Embedding(nuser, d_model)

        self.d_model = d_model

        # --- Output Projection ---
        # nn.Linear is a fully connected layer: output = input @ W^T + bias
        # Input size is 2*d_model because we concatenate the Transformer output
        # (d_model dims) with the user embedding (d_model dims).
        # Output size is ntoken -- one score per movie. These raw scores (logits)
        # are later turned into probabilities via softmax (inside CrossEntropyLoss).
        self.linear = nn.Linear(2 * d_model, ntoken)

        # Initialize all learnable weights to small random values.
        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize embedding and linear layer weights to small uniform random values.

        Why? Neural networks are sensitive to initialization. Starting with very
        large or very small weights can make training unstable (exploding/vanishing
        gradients). Small uniform values [-0.1, 0.1] are a safe starting point.

        Note: The Transformer encoder layers use PyTorch's default initialization
        (Xavier uniform), which is already well-suited for attention layers.
        """
        initrange = 0.1
        # .weight.data gives direct access to the underlying tensor (bypassing autograd)
        self.movie_embedding.weight.data.uniform_(-initrange, initrange)
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()       # biases start at zero
        self.linear.weight.data.uniform_(-initrange, initrange)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """
        Generate a causal (look-ahead) mask of shape [sz, sz].

        This is critical for autoregressive (next-item) prediction:
          - At position 0, the model can only see movie 0.
          - At position 1, it can see movies 0 and 1.
          - At position i, it can see movies 0..i (but NOT i+1, i+2, ...).

        Without this mask, the model would "cheat" by looking at future movies
        when predicting the next one, making it useless at inference time.

        The mask is an upper-triangular matrix filled with -inf:
          [[  0, -inf, -inf],
           [  0,    0, -inf],
           [  0,    0,    0]]

        PyTorch's attention adds this mask to the raw attention scores *before*
        softmax. Since softmax(-inf) = 0, masked positions get zero attention weight,
        effectively making them invisible.

        torch.triu: returns the upper triangle of a matrix.
        diagonal=1: starts the triangle one above the main diagonal, so the
                     diagonal itself (position attending to itself) stays 0 (allowed).
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src: Tensor, user: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Forward pass: takes a batch of movie sequences + user IDs, returns
        predicted scores over all movies at each sequence position.

        Args:
            src:      [batch_size, seq_len] -- integer movie IDs for the watch history.
            user:     [batch_size, 1] -- integer user ID for each sample.
            src_mask: [seq_len, seq_len] -- optional causal mask. Auto-generated if None.

        Returns:
            output: [batch_size, seq_len, ntoken] -- raw logit scores for every movie
                    at every position in the sequence. For next-item prediction, we
                    typically use only the LAST position's scores.

        Step-by-step walkthrough (example: batch=32, seq_len=3, d_model=128, ntoken=3884):
        """
        seq_len = src.size(1)  # e.g., 3

        # --- Step 1: Create causal mask ---
        # Shape: [seq_len, seq_len] e.g., [3, 3]
        # .to(src.device) moves it to GPU if the input is on GPU.
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)

        # --- Step 2: Embed movie IDs into dense vectors ---
        # src: [32, 3] (integer IDs) --> movie_embed: [32, 3, 128] (dense vectors)
        # The embedding lookup replaces each integer with its learned 128-dim vector.
        #
        # Multiply by sqrt(d_model): a scaling trick from the original Transformer paper
        # ("Attention Is All You Need"). The embedding values are small, and the
        # positional encoding values are on the order of [-1, 1]. Scaling up the
        # embeddings ensures they aren't drowned out by the positional signal.
        movie_embed = self.movie_embedding(src) * math.sqrt(self.d_model)

        # user: [32, 1] --> user_embed: [32, 1, 128]
        user_embed = self.user_embedding(user) * math.sqrt(self.d_model)

        # --- Step 3: Add positional encoding ---
        # movie_embed: [32, 3, 128] --> still [32, 3, 128] but now each position
        # has a unique sine/cosine signal added to it, so the model knows movie order.
        movie_embed = self.pos_encoder(movie_embed)

        # --- Step 4: Pass through the Transformer encoder ---
        # This is where the magic happens. Self-attention lets every movie in the
        # sequence exchange information with every other (allowed) movie.
        #
        # For example, if a user watched [Toy Story, Jurassic Park, The Matrix],
        # the attention mechanism learns: "after an action + sci-fi pattern, the
        # user might want more sci-fi."
        #
        # The causal mask (src_mask) ensures position i only attends to positions 0..i.
        # Input:  [32, 3, 128]
        # Output: [32, 3, 128] -- same shape, but now each position's vector is a
        #         context-aware representation that "knows about" the movies before it.
        output = self.transformer_encoder(movie_embed, src_mask)

        # --- Step 5: Incorporate user preferences ---
        # user_embed is [32, 1, 128]. We need to repeat it across all seq_len positions
        # so we can concatenate it with each position's output.
        # .expand(-1, seq_len, -1) repeats along dim=1 without copying memory:
        # [32, 1, 128] --> [32, 3, 128]
        user_embed = user_embed.expand(-1, output.size(1), -1)

        # Concatenate along the last dimension (feature dim):
        # [32, 3, 128] (transformer out) + [32, 3, 128] (user) = [32, 3, 256]
        # This gives the model both "what the sequence suggests" and "who this user is."
        output = torch.cat((output, user_embed), dim=-1)

        # --- Step 6: Project to movie scores ---
        # Linear layer maps [32, 3, 256] --> [32, 3, ntoken] (e.g., [32, 3, 3884])
        # Each of the 3884 values is a raw score (logit) for that movie.
        # Higher score = model thinks this movie is more likely to be watched next.
        # During training, CrossEntropyLoss applies softmax + compares to the true next movie.
        output = self.linear(output)
        return output
