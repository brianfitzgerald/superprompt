import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionEmbed(nn.Module):
    def __init__(self, dim_model, max_len=512) -> None:
        super().__init__()

        # embedding is a matrix of size (max_len, dim_model)
        # for each possible position i, j contains the sinusoid of frequency i / 10000^(2j/dim_model)
        pe = torch.zeros(max_len, dim_model)
        pe.requires_grad = False

        # create a 2D tensor with the position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (
            torch.arange(0, dim_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / dim_model)
        ).exp()

        # for each 2 entries, starting at 0, we get a sin and cos activation
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # get the position embeddings for all tokens up to the current position idx
        return self.pe[:, : x.size(1)]


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=512, dropout=0.1, max_len=512) -> None:
        super().__init__()

        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        embedding_dim = self.token.embedding_dim
        self.position = PositionEmbed(dim_model=embedding_dim, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence)
        x = x + self.position(x)
        x = self.dropout(x)
        return x


# Compute a single attention head
class Attention(nn.Module):
    # matrix multiplication of query and key, then scaled by the square root of the dimension of the query
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, attn_heads, hidden, dropout=0.1) -> None:
        super().__init__()
        assert hidden % attn_heads == 0

        # We assume d_v always equals d_k
        self.d_k = hidden // attn_heads
        self.h = attn_heads

        # linear layers for query, key and value
        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(3)]
        )
        # final linear layer for output
        self.output_linear = nn.Linear(hidden, hidden)

        # attention - performed per batch of queries
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # linear projection from hidden to d_k * h
        # i.e. for each linear layer, we get the query, key and value
        # these represent the linear layer for each head
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # compute attention for all heads in a batch
        x, attention = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # concatenate all heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # apply final linear layer
        return self.output_linear(x)


class SublayerConnection(nn.Module):
    def __init__(self, hidden, dropout) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Feed forward layer, with dropout and GELU activation
class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden, feed_forward_hidden, dropout=0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden, feed_forward_hidden)
        self.w_2 = nn.Linear(feed_forward_hidden, hidden)
        self.dropout = nn.Dropout(p=dropout)
        # gelu is the same as RELU with a slight dip before 0
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(attn_heads, hidden)
        self.feed_forward = PositionwiseFeedForward(
            hidden, feed_forward_hidden, dropout
        )
        self.input_sublayer = SublayerConnection(hidden, dropout)
        self.output_sublayer = SublayerConnection(hidden, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
        max_len=512,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4  # 4 is hyperparameter

        self.embedding = BERTEmbedding(vocab_size, hidden, dropout, max_len)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
                for _ in range(n_layers)
            ]
        )

        # masked LM
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, mask):
        # attention mask for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        print(mask.shape)
        mask = mask.unsqueeze(1)
        print(mask.shape)

        # get the embedding for the input sequence
        x = self.embedding(x)

        for transformer in self.transformer_blocks:
            x = transformer(x, mask)

        # masked LM
        x = self.softmax(self.linear(x))

        return x
