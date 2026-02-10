# This encoder captures the positions of movie interactions in sequences,
# embedding the order information that the Transformer model needs.
import torch
from torch import nn, Tensor
import math

# d_model: The dimensionality of the input embeddings.
# dropout: Dropout probability applied to the embeddings after positional encoding is added.
# max_len: Maximum length of the sequences this module can handle.

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]

        # Scaling term for sinusoidal functions — distributes frequencies across dimensions.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Positional encoding matrix: [1, max_len, d_model] for batch_first format.
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # even indices: sin
        pe[0, :, 1::2] = torch.cos(position * div_term)  # odd indices: cos
        self.register_buffer('pe', pe)  # not updated during training


    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]  # add positional encodings
        return self.dropout(x)
