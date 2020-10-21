import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

import numpy as np

# from .attention import AttentionLayer, Attention
# from .utils import gather_last, mean_pooling, max_pooling
# import random


class CharCNN(nn.Module):
    """
    Class for CH conditioning
    """

    def __init__(
        self,
        n_ch_tokens,
        ch_maxlen,
        ch_emb_size,
        ch_feature_maps,
        ch_kernel_sizes,
        embs,
    ):
        super(CharCNN, self).__init__()
        assert len(ch_feature_maps) == len(ch_kernel_sizes)

        self.n_ch_tokens = n_ch_tokens
        self.ch_maxlen = ch_maxlen
        self.ch_emb_size = ch_emb_size
        self.ch_feature_maps = ch_feature_maps
        self.ch_kernel_sizes = ch_kernel_sizes

        self.feature_mappers = nn.ModuleList()
        for i in range(len(self.ch_feature_maps)):
            reduced_length = self.ch_maxlen - self.ch_kernel_sizes[i] + 1
            self.feature_mappers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=self.ch_feature_maps[i],
                        kernel_size=(self.ch_kernel_sizes[i], self.ch_emb_size),
                    ),
                    nn.Tanh(),
                    nn.MaxPool2d(kernel_size=(reduced_length, 1)),
                )
            )

        self.g = nn.Linear(1536, 128)
        self.embs = embs

    def forward(self, x):
        # x - [batch_size x maxlen]
        bsize, length = x.size()
        assert length == self.ch_maxlen
        x_embs = self.embs(x).view(bsize, 1, self.ch_maxlen, self.ch_emb_size)

        cnn_features = []
        for i in range(len(self.ch_feature_maps)):
            cnn_features.append(self.feature_mappers[i](x_embs).view(bsize, -1))

        return self.g(torch.cat(cnn_features, dim=1))

    def init_ch(self):
        initrange = 0.5 / self.ch_emb_size
        with torch.no_grad():
            nn.init.uniform_(self.embs.weight, -initrange, initrange)
            for name, p in self.feature_mappers.named_parameters():
                if "bias" in name:
                    nn.init.constant_(p, 0)
                elif "weight" in name:
                    nn.init.xavier_uniform_(p)


class LSTM_Encoder(nn.Module):
    def __init__(
        self, embeddings, hidden, num_layers, input_dropout=0.5, output_dropout=0.5,
    ):
        super(LSTM_Encoder, self).__init__()
        self.bidirectional = True
        self.embed = embeddings
        self.hidden = hidden
        self.num_layers = num_layers
        self.encoder = nn.LSTM(
            self.embed.embedding_dim,
            self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.input_dropout = nn.Dropout(input_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, seq, seq_len, initial_state=None):

        embedded_seq = self.embed(seq)
        embedded_seq = self.input_dropout(embedded_seq)

        encoder_input = nn.utils.rnn.pack_padded_sequence(
            embedded_seq, seq_len, batch_first=True, enforce_sorted=False
        )
        if initial_state is not None:
            encoder_hidden, (h_0, c_0) = self.encoder(
                encoder_input,
                (
                    initial_state.unsqueeze(0).repeat(self.num_layers * 2, 1, 1),
                    initial_state.unsqueeze(0).repeat(self.num_layers * 2, 1, 1),
                ),
            )
        else:
            encoder_hidden, (h_0, c_0) = self.encoder(encoder_input)
        encoder_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            encoder_hidden, batch_first=True
        )
        encoder_hidden = self.output_dropout(encoder_hidden)
        final_hidden = mean_pooling(encoder_hidden, seq_len)

        return final_hidden, encoder_hidden


class InputAttention(nn.Module):
    """
    Class for Input Attention conditioning

    From https://github.com/agadetsky/pytorch-definitions/

    """

    def __init__(
        self,
        n_attn_tokens,
        n_attn_embsize,
        n_attn_hid,
        attn_dropout,
        embs,
        sparse=False,
    ):
        super(InputAttention, self).__init__()
        self.n_attn_tokens = n_attn_tokens
        self.n_attn_embsize = n_attn_embsize
        self.n_attn_hid = n_attn_hid
        self.attn_dropout = attn_dropout
        self.sparse = sparse

        self.embs = embs
        self.embs.sparse = sparse
        self.ann = nn.Sequential(
            nn.Dropout(p=self.attn_dropout),
            nn.Linear(in_features=self.n_attn_embsize, out_features=self.n_attn_hid),
            nn.ReLU(),
        )  # maybe use ReLU or other?

        self.a_linear = nn.Linear(
            in_features=self.n_attn_hid, out_features=self.n_attn_embsize
        )

    def forward(self, word, context):
        x_embs = self.embs(word)
        x_embs = x_embs.squeeze(1)
        mask = self.get_mask(context)
        return mask * x_embs

    def get_mask(self, context):
        context_embs = self.embs(context)
        lengths = context != self.embs.padding_idx
        for_sum_mask = lengths.unsqueeze(2).float()
        lengths = lengths.sum(1).float().view(-1, 1)
        logits = self.a_linear((self.ann(context_embs) * for_sum_mask).sum(1) / lengths)
        return F.sigmoid(logits)

    def init_attn(self, freeze=False):
        initrange = 0.5 / self.n_attn_embsize
        with torch.no_grad():
            nn.init.uniform_(self.embs.weight, -initrange, initrange)
            nn.init.xavier_uniform_(self.a_linear.weight)
            nn.init.constant_(self.a_linear.bias, 0)
            nn.init.xavier_uniform_(self.ann[1].weight)
            nn.init.constant_(self.ann[1].bias, 0)
        self.embs.weight.requires_grad = not freeze

    def init_attn_from_pretrained(self, weights, freeze=False):
        self.load_state_dict(weights)
        self.embs.weight.requires_grad = not freeze


class GRU_Decoder(nn.Module):
    def __init__(
        self,
        embeddings,
        hidden,
        repr_hidden_size,
        num_layers,
        encoder_hidden,
        input_dropout=0,
        output_dropout=0,
        teacher_forcing_p=0.5,
        attention="concat",
        mode="TRAIN",
        latent_size=None,
    ):

        super(GRU_Decoder, self).__init__()
        self.MODE = mode
        if mode == "TRAIN":
            self.embeddings = embeddings

            self.input_dropout = nn.Dropout(input_dropout)
            self.output_dropout = nn.Dropout(output_dropout)

            self.repr_hidden_size = repr_hidden_size

            self.hidden_size = hidden

            self.decoder = VDM_Cell(
                self.embeddings.embedding_dim + self.repr_hidden_size,
                self.hidden_size,
                latent_size,
                encoder_hidden,
                self.embeddings.embedding_dim,
                char_size,
            )

            self.teacher_forcing_p = teacher_forcing_p

            self.attn_dim = 300
            # The 2 appears because we will concatenate the decoded vector with the
            # attended decoded vector
            self.attention_layer = Attention(
                encoder_hidden, self.hidden_size, self.attn_dim, type="general"
            )
            self.combo_layer = nn.Linear(
                self.hidden_size + self.attn_dim, self.hidden_size
            )

            self.output_layer = nn.Linear(
                self.hidden_size, self.embeddings.num_embeddings
            )

            self.loss_function = nn.CrossEntropyLoss(
                reduction="sum", ignore_index=self.embeddings.padding_idx
            )
            self.ppl_loss_function = nn.CrossEntropyLoss(
                ignore_index=self.embeddings.padding_idx
            )

        elif mode == "PRETRAIN":
            self.embeddings = embeddings

            self.input_dropout = nn.Dropout(input_dropout)
            self.output_dropout = nn.Dropout(output_dropout)

            self.repr_hidden_size = repr_hidden_size

            self.hidden_size = hidden

            self.representation_layer = nn.Linear(
                self.embeddings.embedding_dim, self.repr_hidden_size
            )
            self.decoder = nn.GRU(
                self.embeddings.embedding_dim + self.repr_hidden_size,
                self.hidden_size,
                batch_first=True,
                num_layers=num_layers,
            )

            self.teacher_forcing_p = teacher_forcing_p

            # The 2 appears because we will concatenate the decoded vector with the
            # attended decoded vector
            self.output_layer = nn.Linear(
                self.hidden_size, self.embeddings.num_embeddings
            )
            self.ppl_loss_function = nn.CrossEntropyLoss(
                ignore_index=self.embeddings.padding_idx
            )

            # self.loss_function = nn.CrossEntropyLoss(
            #     reduction="mean", ignore_index=self.embeddings.padding_idx
            # )

    def forward(
        self,
        representation,
        seq,
        initial_state=None,
        context_batch_mask=None,
        encoder_hidden_states=None,
    ):
        class_name = self.__class__.__name__
        if self.MODE == "PRETRAIN":
            representation = self.representation_layer(self.embeddings(seq).mean(1))
        assert representation.shape == torch.Size([seq.shape[0], self.repr_hidden_size])
        logits = []
        predictions = []
        attention = []
        batch_size, seq_len = seq.shape
        # batch_size, 1
        original_seq = seq
        seq_i = seq[:, 0].unsqueeze(1)
        if self.MODE == "TRAIN":
            word_dropout = Bernoulli(0.75).sample(seq[:, 1:].shape)
            word_dropout = word_dropout.type(torch.LongTensor)
            seq = seq.cpu()
            seq[:, 1:] = seq[:, 1:] * word_dropout
            seq = seq.cuda()
        # 1, batch_size, hidden_x_dirs
        if initial_state is not None:
            decoder_hidden_tuple_i = initial_state.unsqueeze(0)
        else:
            decoder_hidden_tuple_i = None
        # teacher forcing p
        p = random.random()

        self.attention = []

        # we skip the EOS as input for the decoder
        for i in range(seq_len - 1):

            decoder_hidden_tuple_i, logits_i = self.generate(
                seq_i,
                decoder_hidden_tuple_i,
                representation,
                z,
                context,
                emb,
                cnn,
                context_batch_mask,
                encoder_hidden_states,
            )

            # batch_size
            _, predictions_i = logits_i.max(1)

            logits.append(logits_i)
            predictions.append(predictions_i)

            if self.training and p <= self.teacher_forcing_p:
                # batch_size, 1
                seq_i = seq[:, i + 1].unsqueeze(1)
            else:
                # batch_size, 1
                seq_i = predictions_i.unsqueeze(1)
                seq_i = seq_i.cuda()

        # (seq_len, batch_size)
        predictions = torch.stack(predictions, 0)

        # (batch_size, seq_len)
        predictions = predictions.t().contiguous()

        # (seq_len, batch_size, output_size)
        logits = torch.stack(logits, 0)

        # (batch_size, seq_len, output_size)
        logits = logits.transpose(0, 1).contiguous()

        # (batch_size*seq_len, output_size)
        flat_logits = logits.view(batch_size * (seq_len - 1), -1)

        # (batch_size, seq_len)
        labels = original_seq[:, 1:].contiguous()

        # (batch_size*seq_len)
        flat_labels = labels.view(-1)

        loss = self.loss_function(flat_logits, flat_labels)
        log_ppl = F.cross_entropy(
            flat_logits, flat_labels, ignore_index=self.embeddings.padding_idx
        )
        return loss, predictions, log_ppl

    def generate(
        self,
        tgt_batch_sequences_i,
        decoder_hidden_tuple_i,
        representation,
        z,
        context,
        emb,
        cnn,
        context_batch_mask=None,
        encoder_hidden_states=None,
    ):
        """

        :param tgt_batch_i: torch.LongTensor(1, batch_size)
        :param decoder_hidden_tuple_i: tuple(torch.FloatTensor(1, batch_size, hidden_size))
        :param encoder_hidden_states: torch.FloatTensor(batch_size, seq_len, hidden_x_dirs)
        :param src_batch_mask: torch.LongTensor(batch_size, seq_len)
        :param comment_hidden_states: ?
        :param com_batch_mask: ?
        :return:
        """

        # (batch_size, 1, embedding_size)
        emb_tgt_batch_i = self.embeddings(tgt_batch_sequences_i)
        emb_tgt_batch_i = self.input_dropout(emb_tgt_batch_i)

        # (batch_size, 1, hidden_x_dirs) and (1, batch_size, hidden_size)
        decoder_hidden_states_i, decoder_hidden_tuple_i = self.decoder(
            emb_tgt_batch_i, decoder_hidden_tuple_i, z, context, emb, cnn
        )

        # batch_size, hidden_x_dirs
        s_i = decoder_hidden_states_i.squeeze(1)

        if context_batch_mask is not None:
            t_i, attn_i = self.attention_layer.forward(
                s_i, encoder_hidden_states, context_batch_mask
            )

            self.attention.append(attn_i)
            new_s_i = self.combo_layer(torch.cat([s_i, t_i], -1))
        if self.MODE == "PRETRAIN":
            new_s_i = s_i
        new_s_i = self.output_dropout(new_s_i)

        # batch_size, output_size
        logits_i = self.output_layer(new_s_i)

        return decoder_hidden_tuple_i, logits_i


class BoWLoss(nn.Module):
    def __init__(self, latent_size, vocab_size):

        super(BoWLoss, self).__init__()
        self.linear = nn.Linear(latent_size, vocab_size)
        self.log_softmax_fn = nn.LogSoftmax(dim=1)

    def forward(self, batch_latent, batch_labels, batch_labels_mask):
        batch_size, latent_size = batch_latent.shape

        batch_size_, seq_len = batch_labels.shape

        assert batch_size == batch_size_

        # -> batch_size, vocab_size
        batch_logits = self.linear(batch_latent)
        batch_log_probs = self.log_softmax_fn(batch_logits)

        # batch_size, seq_len
        batch_bow_lls = torch.gather(batch_log_probs, 1, batch_labels)

        masked_batch_bow_lls = batch_bow_lls * batch_labels_mask

        batch_bow_ll = masked_batch_bow_lls.sum(1)

        batch_bow_nll = -batch_bow_ll.sum()

        return batch_bow_nll


class LSTMWordAttention(nn.Module):
    def __init__(
        self, embeddings, hidden_size, dropout=0.5,
    ):
        super(LSTMWordAttention, self).__init__()

        self.embeddings = embeddings

        self.hidden = hidden_size

        self.f = nn.Linear(hidden_size * 2 + embeddings.embedding_dim, hidden_size * 2)
        self.encoder = LSTM_Encoder(
            embeddings, hidden_size, 2, input_dropout=0.5, output_dropout=0.5,
        )

        self.s = nn.Linear(self.embeddings.embedding_dim, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, word, context, context_lengths, index_of_word):

        e_word = self.embeddings(word)

        mean_hidden, encoder_hidden = self.encoder(
            context, context_lengths, initial_state=self.s(e_word)
        )
        word_repr = encoder_hidden[torch.arange(mean_hidden.shape[0]), index_of_word]

        meta = torch.cat((word_repr.unsqueeze(1), mean_hidden.unsqueeze(1)), 1).mean(1)
        out = self.dropout(self.f(torch.cat([meta, e_word], -1)))
        return meta, encoder_hidden
