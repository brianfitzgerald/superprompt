import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    # src = [src len, batch size]
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions]
        # n_directions is 1 when we use LSTM since we are predicting next token only
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # cell is the cell state for each layer
        # cell state is similar to residual - it is the information that is passed to the next layer without being
        # passed to the output
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    # input = [batch size]
    # hidden = [n layers * n directions, batch size, hid dim]
    # cell = [n layers * n directions, batch size, hid dim]
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        pred = self.fc_out(output.squeeze(0))

        return pred, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(
        self, encoder: Encoder, decoder: Decoder, device: torch.device
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        # first input is <sos> token
        input = trg[0, :]

        for t in range(1, trg_len):
            # each iteration, we predict one token, and add that to the output tensor
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            max_pred = output.argmax(1)

            # input for next iteration is the actual token if teacher forcing is on, otherwise it is the predicted token
            input = trg[t] if teacher_force else max_pred

        return outputs
