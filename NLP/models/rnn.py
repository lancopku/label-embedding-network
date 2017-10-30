import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict
import models


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        return outputs, state



class rnn_decoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None, score_fn=None):
        super(rnn_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)

        self.score_fn = score_fn
        self.linear = nn.Linear(config.hidden_size, vocab_size)

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None

        self.attention = models.global_attention(config.hidden_size, activation)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, inputs, init_state, contexts):
        embs = self.embedding(inputs)
        outputs, state, attns = [], init_state, []
        for emb in embs.split(1):
            output, state = self.rnn(emb.squeeze(0), state)
            output, attn_weights = self.attention(output, contexts)
            output = self.dropout(output)
            outputs += [output]
            attns += [attn_weights]
        outputs = torch.stack(outputs)
        attns = torch.stack(attns)
        #outputs = self.linear(outputs.view(-1, self.hidden_size))
        return outputs, state

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores

    def sample(self, input, init_state, contexts):
        #emb = self.embedding(input)
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = self.config.max_tgt_len

        for i in range(max_time_step):
            output, state, attn_weights = self.sample_one(inputs[i], state, contexts)
            predicted = output.max(1)[1]
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]

        sample_ids = torch.stack(sample_ids)
        attns = torch.stack(attns)
        return sample_ids, (outputs, attns)

    def sample_one(self, input, state, contexts):
        emb = self.embedding(input)
        output, state = self.rnn(emb, state)
        hidden, attn_weigths = self.attention(output, contexts)
        output = self.compute_score(hidden)

        return output, state, attn_weigths
