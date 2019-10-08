from enum import Enum

import torch.nn as nn
import torch
import torch.nn.functional as f


class EmbeddingDimension(Enum):
    d300 = 300
    d200 = 200
    d100 = 100
    d50 = 50


class EncoderRNN(nn.Module):
    """

    """
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set
        # to 'hidden_size' because our input size is a word embedding with
        # number of features == hidden_size
        self.gru = nn.GRU(input_size=embedding.embedding_dim,
                          hidden_size=hidden_size, num_layers=n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """

        :param input_seq:
        :param input_lengths:
        :param hidden:
        :return:
        """
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :,
                                                     self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attention(nn.Module):
    """

    """

    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                             "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        """

        :param hidden:
        :param encoder_output:
        :return:
        """
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        """

        :param hidden:
        :param encoder_output:
        :return:
        """
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        """

        :param hidden:
        :param encoder_output:
        :return:
        """
        energy = self.attn(torch.cat(
            (hidden.expand(encoder_output.size(0), -1, -1), encoder_output),
            2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        """

        :param hidden:
        :param encoder_outputs:
        :return:
        """
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attention_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attention_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attention_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attention_energies = attention_energies.t()

        # Return the softmax normalized probability scores (with added
        # dimension)
        return f.softmax(attention_energies, dim=1).unsqueeze(1)


class LuongAttentionDecoderRNN(nn.Module):
    """ Implements the Luong et al. (2015) attention decoder layer.
    """

    def __init__(self, attention_model, embedding, hidden_size, output_size,
                 number_of_layers=1, dropout=0.1):
        super(LuongAttentionDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attention_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = number_of_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding.embedding_dim, hidden_size,
                          number_of_layers,
                          dropout=(0 if number_of_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attention(attention_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """

        :param input_step:
        :param last_hidden:
        :param encoder_outputs:
        :return:
        """
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attention_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted
        # sum" context vector
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = f.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


