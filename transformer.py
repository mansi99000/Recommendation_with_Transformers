import torch
from torch import nn, Tensor
from positional_encoder import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# whenever you write self.attribute or self.method(), 
# Python knows you are referring to an instance variable or method, 
# rather than a local variable or an external function.

class TransformerModel(nn.Module):
    """ 
    ntoken: Number of unique tokens (i.e., size of the movie vocabulary).
    nuser: Number of users.
    d_model: Dimension of the model, i.e., size of the embeddings.
    nhead: Number of heads in the multi-head attention mechanism.
    d_hid: Dimension of the hidden feedforward network inside the transformer encoder.
    nlayers: Number of transformer encoder layers.
    dropout: Dropout rate, with a default value of 0.5. 
    """

    def __init__(self, ntoken: int, nuser: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'  # self.something is associated with the instance of the class # self stores data and methods inside each instance
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Why
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.movie_embedding = nn.Embedding(ntoken, d_model)
        self.user_embedding = nn.Embedding(nuser, d_model)

        self.d_model = d_model
        self.linear = nn.Linear(2*d_model, ntoken) # WHy

        self.init_weights()

    def init_weights(self) -> None:
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











