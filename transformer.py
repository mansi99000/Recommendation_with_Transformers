import math

import torch
from torch import nn, Tensor
from positional_encoder import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    """
    Transformer-based sequential recommendation model.

    ntoken: Number of unique tokens (size of the movie vocabulary).
    nuser: Number of users.
    d_model: Dimension of the model (embedding size).
    nhead: Number of heads in multi-head attention.
    d_hid: Dimension of the hidden feedforward network.
    nlayers: Number of transformer encoder layers.
    dropout: Dropout rate (default 0.5).
    """

    def __init__(self, ntoken: int, nuser: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # batch_first=True so input/output shape is [batch, seq_len, d_model]
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.movie_embedding = nn.Embedding(ntoken, d_model)
        self.user_embedding = nn.Embedding(nuser, d_model)

        self.d_model = d_model
        self.linear = nn.Linear(2 * d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.movie_embedding.weight.data.uniform_(-initrange, initrange)
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generate an upper-triangular causal mask to prevent attending to future positions.

        This ensures that predictions at position i can only depend on positions 0..i,
        which is critical for autoregressive next-item prediction.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src: Tensor, user: Tensor, src_mask: Tensor = None) -> Tensor:
        seq_len = src.size(1)  # src: [batch, seq_len]

        # Generate causal mask if not provided
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)

        # Embedding movie ids and user id
        movie_embed = self.movie_embedding(src) * math.sqrt(self.d_model)
        user_embed = self.user_embedding(user) * math.sqrt(self.d_model)

        # Positional encoding
        movie_embed = self.pos_encoder(movie_embed)

        # Transformer encoder with causal mask prevents future information leakage
        output = self.transformer_encoder(movie_embed, src_mask)

        # Expand user_embed along the sequence length dimension
        user_embed = user_embed.expand(-1, output.size(1), -1)

        # Concatenate user embeddings with transformer output
        output = torch.cat((output, user_embed), dim=-1)

        output = self.linear(output)
        return output
