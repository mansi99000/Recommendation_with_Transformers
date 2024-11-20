# This encoder will capture the positions of movie interactions in our sequences,
#  thus embedding the order information that the Transformer model needs.
import torch
from torch import nn, Tensor
import math

# d_model: The dimensionality of the input embeddings.
# dropout: Dropout probability applied to the embeddings after positional encoding is added.
# max_len: Maximum length of the sequences this module can handle. Precomputes positional encodings up to this length.

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000): # Constructor
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # creates a tensor of with no. of rows = max_len and column = 1

        # `div_term` is used in the calculation of the sinusoidal values.
        # A scaling term for the sinusoidal functions, designed to distribute frequencies across dimensions. 
        # The exponential ensures a wide range of wavelengths.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Initializing positional encoding matrix with zeros.
        pe = torch.zeros(max_len, 1, d_model) # We can scale the vector in each dimension, so our pe has the dimesion of the input

        # Slicing notation:
        # 0::2 start at 0 and step by 2; meaning select every second term
        # tensor[start:stop:step]
        # pe[:, 0, 0::2] = select all rows in teh first dimension; select the first elemnet in teh second dimension (it only has 1 column); and perform slicing in teh pe dimension
        # Calculating the positional encodings.
        pe[:, 0, 0::2] = torch.sin(position * div_term) # for even indices it applies sin
        pe[:, 0, 1::2] = torch.cos(position * div_term) # for odd indices it applies cos
        self.register_buffer('pe', pe) # this ensure that positional encodings are part of the module state and are not updated during training


    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)] # Adds positional encodings to the input embeddings
        return self.dropout(x) # Applies dropout to the final output for regularization