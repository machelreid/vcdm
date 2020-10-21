import transformers
from allennlp.modules.elmo import Elmo
from transformers import AutoConfig, AutoModel
from torch.distributions.bernoulli import Bernoulli
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention as AttentionLayer
import random

__ELMO_OPTIONS__ = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
__ELMO_WEIGHTS__ = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"


class ELMo_wrapper(nn.Module):
    def __init__(self):
        super(ELMo_wrapper, self).__init__()

        self.model = Elmo(options_file, weight_file, 2, dropout=0)

    def forward(self, input_ids, lens):
        embeddings = self.model(input_ids)
        avg_embeddings = self._avg_pool(embeddings, lens)
        return (embeddings, avg_embeddings, embeddings)

    def _avg_pool(self, embeddings, lens):
        _sum = torch.sum(embeddings, 1)
        assert lens.shape[0] == embeddings.shape[0]
        assert lens.shape[1] == 1
        return _sum / lens


def get_pretrained_transformer(path):
    config = AutoConfig.from_pretrained(path, output_hidden_states=True)
    return AutoModel.from_pretrained(path, config=config)


class LSTM_Encoder(nn.Module):
    def __init__(self, embeddings, hidden, layers, dropout, word_dropout=None):
        super(LSTM_Encoder, self).__init__()
        self.embeddings = embeddings
        self.hidden = hidden
        self.num_layers = layers

        self.encoder = nn.LSTM(
            self.embeddings.embedding_dim,
            self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.word_dropout_p = word_dropout
        self.input_dropout = nn.Dropout(dropout.input)
        self.output_dropout = nn.Dropout(dropout.output)

    def forward(self, input, lens, init_state=None, aggregator="mean"):
        if self.word_dropout is not None:
            input = self.word_dropout(input, lens)
        embedded_seq = self.embeddings(input)
        embedded_seq = self.input_dropout(embedded_seq)

        encoder_input = nn.utils.rnn.pack_padded_sequence(
            embedded_seq, lens, batch_first=True, enforce_sorted=False
        )
        if init_state is not None:
            encoder_hidden, (h_0, c_0) = self.encoder(
                encoder_input,
                (
                    init_state.unsqueeze(0).repeat(self.num_layers * 2, 1, 1),
                    init_state.unsqueeze(0).repeat(self.num_layers * 2, 1, 1),
                ),
            )
        else:
            encoder_hidden, (h_0, c_0) = self.encoder(encoder_input)
        encoder_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            encoder_hidden, batch_first=True
        )
        if aggregator == "last":
            h_0 = self.output_dropout(h_0)
            return (
                torch.cat(
                    h_0.view(self.num_layers, 2, input.shape[0], self.hidden)[-1].chunk(
                        2
                    ),
                    -1,
                ).squeeze(0),
                encoder_hidden,
            )
        elif aggregator == "mean":
            encoder_hidden = self.output_dropout(encoder_hidden)
            return encoder_hidden.sum(1) / lens.unsqueeze(1), encoder_hidden
        elif aggregator == "max":
            encoder_hidden = self.output_dropout(encoder_hidden)
            return (
                torch.stack([h[:l].max(0)[0] for h, l in zip(encoder_hidden, lens)], 0),
                encoder_hidden,
            )
        else:
            raise NotImplementedError(
                f"Aggregator `{aggregator}` not in ['last','max','mean']"
            )

    def word_dropout(self, input, lens):
        if not self.training:
            return input
        output = []
        for inp, _len in zip(input, lens):
            word_dropout = Bernoulli(1 - self.word_dropout_p).sample(inp[1:_len].shape)
            inp = inp.cpu()
            inp[1:_len] = inp[1:_len] * word_dropout.type(torch.LongTensor)
            inp[1:_len][inp[1:_len] == 0] = self.embeddings.unk_idx
            inp = inp.cuda()
            output.append(inp)

        return torch.stack(output, 0).cuda()
