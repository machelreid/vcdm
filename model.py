import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.span_extractors import (
    EndpointSpanExtractor,
    SelfAttentiveSpanExtractor,
)
from allennlp.modules.scalar_mix import ScalarMix
from utils import sequence_mask, find_subtensor, batched_span_select
from dotmap import DotMap
import random
from beam import BeamSearch
from onmt.modules import GlobalAttention
from onmt.translate import GNMTGlobalScorer
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from attention import Attention


class DefinitionProbing(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_pretrained,
        encoder_frozen,
        decoder_hidden,
        embeddings,
        max_layer=12,
        src_pad_idx=0,
        encoder_hidden=None,
        variational=None,
        latent_size=None,
        scalar_mix=False,
        aggregator="mean",
        teacher_forcing_p=0.3,
        classification=None,
        attentional=False,
        definition_encoder=None,
        word_dropout_p=None,
        decoder_num_layers=None,
    ):
        super(DefinitionProbing, self).__init__()

        self.embeddings = embeddings
        self.variational = variational
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.decoder_num_layers = decoder_num_layers
        self.encoder = encoder
        self.latent_size = latent_size
        self.src_pad_idx = src_pad_idx
        if encoder_pretrained:
            self.encoder_hidden = self.encoder.config.hidden_size
        if encoder_frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.max_layer = max_layer
        self.aggregator = aggregator
        if self.aggregator == "span":
            self.span_extractor = SelfAttentiveSpanExtractor(self.encoder_hidden)
        self.context_feed_forward = nn.Linear(self.encoder_hidden, self.encoder_hidden)
        self.scalar_mix = None
        if scalar_mix:
            self.scalar_mix = ScalarMix(self.max_layer + 1)
        self.global_scorer = GNMTGlobalScorer(
            alpha=2, beta=None, length_penalty="avg", coverage_penalty=None
        )

        self.decoder = LSTM_Decoder(
            embeddings.tgt,
            hidden=self.decoder_hidden,
            encoder_hidden=self.encoder_hidden,
            num_layers=self.decoder_num_layers,
            word_dropout=word_dropout_p,
            teacher_forcing_p=teacher_forcing_p,
            attention="general" if attentional else None,
            dropout=DotMap({"input": 0.5, "output": 0.5}),
            decoder="VDM" if self.variational else "LSTM",
            variational=self.variational,
            latent_size=self.latent_size,
        )

        self.target_kl = 1.0
        if self.variational:
            self.definition_encoder = definition_encoder
            self.definition_feed_forward = nn.Linear(
                self.encoder_hidden, self.encoder_hidden
            )
            self.mean_layer = nn.Linear(self.latent_size, self.latent_size)
            self.logvar_layer = nn.Linear(self.latent_size, self.latent_size)
            self.w_z_post = nn.Sequential(
                nn.Linear(self.encoder_hidden * 2, self.latent_size), nn.Tanh()
            )
            self.mean_prime_layer = nn.Linear(self.latent_size, self.latent_size)
            self.logvar_prime_layer = nn.Linear(self.latent_size, self.latent_size)
            self.w_z_prior = nn.Sequential(
                nn.Linear(self.encoder_hidden, self.latent_size), nn.Tanh()
            )
            self.z_project = nn.Sequential(
                nn.Linear(self.latent_size, self.decoder_hidden), nn.Tanh()
            )

    def forward(
        self,
        input,
        seq_lens,
        span_token_ids,
        target,
        target_lens,
        definition=None,
        definition_lens=None,
        classification_labels=None,
        sentence_mask=None,
    ):
        batch_size, tgt_len = target.shape

        # (batch_size,seq_len,hidden_size), (batch_size,hidden_size), (num_layers,batch_size,seq_len,hidden_size)
        last_hidden_layer, sentence_representation, all_hidden_layers = self.encoder(
            input, attention_mask=sequence_mask(seq_lens), token_type_ids=sentence_mask
        )

        cosine_loss = None
        loss = None
        KLD = None
        fake_loss_kl = None
        if self.aggregator == "cls":
            cls_hidden = last_hidden_layer[:, 0, :].squeeze(1)
            span_representation = self.context_feed_forward(cls_hidden)
            hidden_states = last_hidden_layer

        else:
            span_ids = self._id_extractor(
                tokens=span_token_ids, batch=input, lens=seq_lens
            )

            span_representation, hidden_states = self._span_aggregator(
                all_hidden_layers if self.scalar_mix is not None else last_hidden_layer,
                sequence_mask(seq_lens),
                span_ids,
            )
            span_representation = self.context_feed_forward(span_representation)
        if self.variational:
            (
                definition_last_hidden_layer,
                _,
                definition_all_hidden_layers,
            ) = self.definition_encoder(
                definition, attention_mask=sequence_mask(definition_lens)
            )
            definition_representation = self.definition_feed_forward(
                definition_last_hidden_layer[:, 0]
            )

            post_project = self.w_z_post(
                torch.cat([span_representation, definition_representation], -1)
            )
            prior_project = self.w_z_prior(span_representation)

            mu = self.mean_layer(post_project)
            logvar = self.logvar_layer(post_project)

            mu_prime = self.mean_prime_layer(prior_project)
            logvar_prime = self.logvar_prime_layer(prior_project)

            z = mu + torch.exp(logvar * 0.5) * torch.randn_like(logvar)
            span_representation = self.z_project(z)
            KLD = kl_divergence(
                Normal(mu, torch.exp(logvar * 0.5)),
                Normal(mu_prime, torch.exp(logvar_prime * 0.5)),
            )
            kl_mask = (KLD > (self.target_kl / self.latent_size)).float()
            fake_loss_kl = (kl_mask * KLD).sum(dim=1)

        predictions, logits = self.decoder(
            target, target_lens, span_representation, hidden_states, seq_lens,
        )

        if self.variational:
            loss = (
                F.cross_entropy(
                    logits.view(batch_size * (tgt_len - 1), -1),
                    target[:, 1:].contiguous().view(-1),
                    ignore_index=self.embeddings.tgt.padding_idx,
                    reduction="none",
                )
                .view(batch_size, tgt_len - 1)
                .sum(1)
            )

            perplexity = F.cross_entropy(
                logits.view(batch_size * (tgt_len - 1), -1),
                target[:, 1:].contiguous().view(-1),
                ignore_index=self.embeddings.tgt.padding_idx,
                reduction="mean",
            ).exp()
        else:
            loss = F.cross_entropy(
                logits.view(batch_size * (tgt_len - 1), -1),
                target[:, 1:].contiguous().view(-1),
                ignore_index=self.embeddings.tgt.padding_idx,
            )
            perplexity = loss.exp()
        return DotMap(
            {
                "predictions": predictions,
                "logits": logits,
                "loss": loss,
                "perplexity": perplexity,
                "fake_kl": fake_loss_kl,
                "kl": KLD,
                "cosine_loss": cosine_loss,
            }
        )

    def _validate(
        self,
        input,
        seq_lens,
        span_token_ids,
        target,
        target_lens,
        decode_strategy,
        definition=None,
        definition_lens=None,
        sentence_mask=None,
    ):
        batch_size, tgt_len = target.shape

        # (batch_size,seq_len,hidden_size), (batch_size,hidden_size), (num_layers,batch_size,seq_len,hidden_size)
        last_hidden_layer, pooled_representation, all_hidden_layers = self.encoder(
            input, attention_mask=sequence_mask(seq_lens), token_type_ids=sentence_mask
        )

        KLD = None
        mu_prime = None
        if self.aggregator == "cls":
            cls_hidden = last_hidden_layer[:, 0]
            cls_hidden_forwarded = self.cls_feed_forward(cls_hidden)
            span_representation = cls_hidden_forwarded
            hidden_states = last_hidden_layer
        else:
            span_ids = self._id_extractor(
                tokens=span_token_ids, batch=input, lens=seq_lens
            )

            span_representation, hidden_states = self._span_aggregator(
                all_hidden_layers if self.scalar_mix is not None else last_hidden_layer,
                sequence_mask(seq_lens),
                span_ids,
            )

            span_representation = self.context_feed_forward(span_representation)
        if self.variational:
            (
                definition_last_hidden_layer,
                _,
                definition_all_hidden_layers,
            ) = self.definition_encoder(
                definition, attention_mask=sequence_mask(definition_lens)
            )
            definition_representation = self.definition_feed_forward(
                definition_last_hidden_layer[:, 0]
            )

            post_project = self.w_z_post(
                torch.cat([span_representation, definition_representation], -1)
            )
            prior_project = self.w_z_prior(span_representation)

            mu = self.mean_layer(post_project)
            logvar = self.logvar_layer(post_project)

            mu_prime = self.mean_prime_layer(prior_project)
            logvar_prime = self.logvar_prime_layer(prior_project)

            hidden_states = last_hidden_layer

            KLD = (
                kl_divergence(
                    Normal(mu, torch.exp(logvar * 0.5)),
                    Normal(mu_prime, torch.exp(logvar_prime * 0.5)),
                )
                .sum(1)
                .mean()
            )

            span_representation = self.z_project(mu_prime)
        memory_bank = hidden_states if self.decoder.attention else None
        _, logits = self.decoder(
            target, target_lens, span_representation, memory_bank, seq_lens,
        )

        loss = F.cross_entropy(
            logits.view(batch_size * (tgt_len - 1), -1),
            target[:, 1:].contiguous().view(-1),
            ignore_index=self.embeddings.tgt.padding_idx,
        )

        ppl = loss.exp()
        beam_results = self._strategic_decode(
            target,
            target_lens,
            decode_strategy,
            memory_bank,
            seq_lens,
            span_representation,
        )
        return DotMap(
            {
                "predictions": beam_results["predictions"],
                "logits": logits.view(batch_size * (tgt_len - 1), -1),
                "loss": loss,
                "perplexity": ppl,
                "kl": KLD,
            }
        )

    def _strategic_decode(
        self,
        target,
        target_lens,
        decode_strategy,
        memory_bank,
        memory_lengths,
        span_representation,
    ):
        """Translate a batch of sentences step by step using cache.
        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
        generate translation step by step.
        Returns:
            results (dict): The translation results.
        """

        parallel_paths = decode_strategy.parallel_paths  # beam_size

        # (0) Prep the components of the search.
        # use_src_map = self.copy_attn
        batch_size, max_len = target.shape
        # Initialize the hidden states
        self.decoder.init_state(
            span_representation, encoder_hidden=memory_bank, src_lens=memory_lengths
        )

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            # "gold_score": self._gold_score(
            #    batch, memory_bank, src_lengths, src_vocabs, use_src_map,
            #    enc_states, batch_size, src)
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None  # batch.src_map if use_src_map else None
        fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(
            memory_bank, memory_lengths, src_map, device="cuda"
        )
        if fn_map_state is not None:
            self.decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions

            logits, attn = self.decoder.generate(
                decoder_input, memory_bank, memory_lengths
            )

            decode_strategy.advance(F.log_softmax(logits, 1), attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if memory_bank is not None:
                    if isinstance(memory_bank, tuple):
                        memory_bank = tuple(
                            x.index_select(0, select_indices) for x in memory_bank
                        )
                    else:
                        memory_bank = memory_bank.index_select(0, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                self.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        # if self.decoder.attention:
        #    results["alignment"] = self._align_forward(
        #        batch, decode_strategy.predictions
        #    )
        # else:
        #    results["alignment"] = [[] for _ in range(batch_size)]
        return results

    def _span_aggregator(
        self, hidden_states, input_mask, span_ids, layer_no: int = None,
    ):

        if layer_no is not None:
            hidden_states = hidden_states[layer_no]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[: self.max_layer + 1]
            hidden_states = self.scalar_mix(hidden_states, mask=input_mask)
        if self.aggregator == "span":
            span = self.span_extractor(hidden_states, span_ids).squeeze(
                1
            )  # As we will only be extracting one span per sequence
        elif self.aggregator == "mean":
            extracted_embeddings, span_mask = batched_span_select(
                hidden_states, span_ids
            )
            extracted_embeddings = (
                extracted_embeddings * span_mask.type(torch.float).unsqueeze(-1)
            ).squeeze(1)
            lengths = (span_mask == True).sum(-1) + 1e-20  # To avoid zero division
            span = extracted_embeddings.sum(1) / lengths
        return span, hidden_states

    def _id_extractor(self, tokens, batch, lens):
        """
        Extracts span indices given a sequence, if none found returns the span as the start and end of sequence as the span
        """
        with torch.no_grad():
            output_ids = []
            for w in tokens:
                output_ids.append(
                    torch.tensor(
                        list(filter((self.src_pad_idx).__ne__, w.tolist()))[1:-1]
                    )
                )

            output_indices = []
            for i in range(batch.shape[0]):
                tensor = find_subtensor(output_ids[i], batch[i])
                if tensor is None:
                    tensor = torch.tensor([1, lens[i].item() - 1]).to("cuda")
                output_indices.append(tensor)
            return torch.stack(output_indices).unsqueeze(1)


class LSTM_Decoder(nn.Module):
    def __init__(
        self,
        embeddings=None,
        hidden=None,
        encoder_hidden=None,
        num_layers=1,
        teacher_forcing_p=0.0,
        attention=None,
        dropout=None,
        char_encoder_hidden=None,
        char_attention=None,
        word_dropout=None,
        encoder_feeding=False,
        variational=False,
        decoder="LSTM",
        latent_size=None,
    ):

        super(LSTM_Decoder, self).__init__()
        # TODO Use Fairseq attention
        self.embeddings = embeddings
        self.hidden = hidden
        self.embedding_dropout = nn.Dropout(dropout.input)

        self.hidden_state_dropout = nn.Dropout(dropout.output)
        self.word_dropout_p = word_dropout if word_dropout is not None else 0.0
        self.variational = variational
        if self.variational:
            encoder_feeding = True
            self.encoder_hidden_proj = lambda x: x
        else:
            self.encoder_hidden_proj = (
                nn.Linear(encoder_hidden, hidden, bias=False)
                if (encoder_hidden != hidden or encoder_hidden is None)
                else lambda x: x
            )

        self.lstm_decoder = nn.ModuleList()
        self.num_layers = num_layers

        self.input_feeding_size = 0  # latent_size if self.variational else 0

        if decoder == "LSTM":
            decoder_cell = [nn.LSTMCell] * self.num_layers
        elif decoder == "VDM":
            if self.num_layers == 1:
                decoder_cell = [VDM_LSTMCell]
            else:
                decoder_cell = [nn.LSTMCell] * (self.num_layers - 1) + [VDM_LSTMCell]
        for i in range(self.num_layers):
            if i == 0:
                self.lstm_decoder.append(
                    decoder_cell[i](
                        self.embeddings.embedding_dim + self.input_feeding_size,
                        self.hidden,
                    )
                )
            else:
                self.lstm_decoder.append(decoder_cell[i](self.hidden, self.hidden))
        self.teacher_forcing_p = teacher_forcing_p
        self.state = {
            "hidden": [None] * self.num_layers,
            "cell": [None] * self.num_layers,
            "latent": [None] * self.num_layers,
        }
        self.attention = attention
        self.proj_layer = nn.Linear(self.hidden, self.embeddings.num_embeddings)
        if attention is not None:
            self.enc_hidden_att_komp = (
                nn.Linear(encoder_hidden, hidden, bias=False)
                if (encoder_hidden != hidden or encoder_hidden is None)
                else lambda x: x
            )
            self.attention = Attention(
                self.hidden, attn_type=attention, attn_func="softmax",
            )
        self.char_attention = char_attention
        if char_attention is not None:
            self.char_hidden_att_komp = (
                nn.Linear(char_encoder_hidden, hidden, bias=False)
                if (char_encoder_hidden != hidden or char_encoder_hidden is None)
                else lambda x: x
            )
            self.char_attention = GlobalAttention(
                self.hidden, attn_type=attention, attn_func="softmax"
            )
            self.proj_layer = nn.Linear(self.hidden * 2, self.embeddings.num_embeddings)

    def forward(
        self,
        input_ids,
        lens=None,
        initial_state=None,
        encoder_hidden=None,
        src_lens=None,
        character_encoder_hidden=None,
        character_lens=None,
    ):
        self.init_state(initial_state, encoder_hidden, src_lens)
        input_ids = self.word_dropout(input_ids, lens)
        all_logits = []
        all_preds = []

        for i in range(input_ids.shape[1] - 1):
            p = random.random()

            input_id = (
                all_preds[-1]
                if (p <= self.teacher_forcing_p and all_preds and self.training)
                else input_ids[:, i]
            )
            logits, attn = self.generate(
                input_id,
                encoder_hidden,
                src_lens,
                character_encoder_hidden,
                character_lens,
            )
            all_logits.append(logits)

            pred = torch.argmax(F.softmax(logits, 1), 1)
            all_preds.append(pred)

        # batch_size, seq_len
        all_logits = torch.stack(all_logits, 1).contiguous()
        all_preds = torch.stack(all_preds, 1).contiguous()
        return all_preds, all_logits

    def generate(
        self,
        input_id,
        encoder_hidden=None,
        src_lens=None,
        character_encoder_hidden=None,
        char_lens=None,
    ):
        input = self.embedding_dropout(self.embeddings(input_id))

        for i, rnn in enumerate(self.lstm_decoder):
            if isinstance(rnn, VDM_LSTMCell):
                # vdm lstm cell
                hidden, cell = rnn(
                    input,
                    (
                        self.state["hidden"][i],
                        self.state["cell"][i],
                        self.latent,
                        self.context,
                    ),
                )
            else:
                # recurrent cell
                hidden, cell = rnn(
                    input, (self.state["hidden"][i], self.state["cell"][i])
                )

            # hidden state becomes the input to the next layer
            input = self.hidden_state_dropout(hidden)

            # save state for next time step
            self.state["hidden"][i] = hidden
            self.state["cell"][i] = cell

        out = hidden
        _att = None
        if self.attention is not None:
            _enc = self.enc_hidden_att_komp(encoder_hidden)
            _hidden = hidden
            out, _att, self.context = self.attention(_hidden, _enc, src_lens)
        if self.char_attention is not None:
            _enc = self.char_hidden_att_komp(character_encoder_hidden)
            _hidden = hidden
            hidden, att = self.char_attention(_hidden, _enc, char_lens)
            out = torch.cat([out, hidden], -1)

        logits = self.proj_layer(out)
        return logits, _att

    def init_state(self, initial_state, encoder_hidden=None, src_lens=None):
        if initial_state is None:
            self.state["hidden"] = [None] * self.num_layers
            self.state["cell"] = [None] * self.num_layers
        else:
            self.state["hidden"] = [
                self.encoder_hidden_proj(initial_state)
            ] * self.num_layers
            self.state["cell"] = [
                self.encoder_hidden_proj(initial_state)
            ] * self.num_layers
            if self.variational:
                self.latent = initial_state  # latent_variable
                self.context = self.attention(
                    initial_state, self.enc_hidden_att_komp(encoder_hidden), src_lens
                )[2]

    def map_state(self, fn):
        self.state["hidden"] = [fn(h, 0) for h in self.state["hidden"]]
        self.state["cell"] = [fn(c, 0) for c in self.state["cell"]]
        if self.variational:
            self.latent = fn(self.latent, 0)
            self.context = fn(self.context, 0)

    def word_dropout(self, input, lens):
        if not self.training:
            return input
        output = []
        for inp, _len in zip(input, lens):
            word_dropout = Bernoulli(1 - self.word_dropout_p).sample(
                inp[1 : _len - 1].shape
            )
            inp = inp.cpu()
            inp[1 : _len - 1] = inp[1 : _len - 1] * word_dropout.type(torch.LongTensor)
            inp[1 : _len - 1][inp[1 : _len - 1] == 0] = self.embeddings.unk_idx
            inp = inp.cuda()
            output.append(inp)

        return torch.stack(output, 0).cuda()


class VDM_LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VDM_LSTMCell, self).__init__()
        self.w_i = nn.Linear(input_dim, hidden_dim * 4)
        self.w_h = nn.Linear(hidden_dim, hidden_dim * 4)
        self.w_z = nn.Linear(hidden_dim, hidden_dim * 4)
        self.w_c = nn.Linear(hidden_dim, hidden_dim * 4)

    def forward(self, input, hidden):
        hx, cx, zx, context_x = hidden

        gates = self.w_i(input) + self.w_h(hx) + self.w_z(zx) + self.w_c(context_x)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


class VDM_GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VDM_LSTMCell, self).__init__()
        self.w_i = nn.Linear(input_dim, hidden_dim * 3)
        self.w_h = nn.Linear(hidden_dim, hidden_dim * 3)
        self.w_z = nn.Linear(hidden_dim, hidden_dim * 3)
        self.w_c = nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, input, hidden):
        hx, zx, cx = hidden
        gi = self.w_i(input)
        gh = self.w_h(hx)
        gz = self.w_z(zx)
        gc = self.w_c(cx)

        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_h, h_n = gh.chunk(3, 1)
        z_r, z_z, z_n = gz.chunk(3, 1)
        c_r, c_c, c_n = gc.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r + z_r + c_r)
        inputgate = torch.sigmoid(i_i + h_i + z_i + c_i)
        newgate = torch.tanh(i_n + (resetgate * h_n) + c_n + z_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy, cy


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
        h_0 = self.output_dropout(h_0)
        final_hidden = (
            torch.cat(
                h_0.view(self.num_layers, 2, seq.shape[0], self.hidden)[-1].chunk(2),
                -1,
            ).squeeze(0),
            encoder_hidden,
        )
        return final_hidden, encoder_hidden


class DefinitionProbingLSTM(nn.Module):
    def __init__(
        self,
        encoder,
        decoder_hidden,
        embeddings,
        max_layer=12,
        src_pad_idx=0,
        encoder_hidden=None,
        latent_size=None,
        scalar_mix=False,
        aggregator="mean",
        teacher_forcing_p=0.3,
        classification=None,
        attentional=False,
        definition_encoder=None,
        word_dropout_p=None,
        decoder_num_layers=2,
    ):
        super(DefinitionProbingLSTM, self).__init__()

        self.embeddings = embeddings
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.decoder_num_layers = decoder_num_layers
        self.encoder = encoder
        self.latent_size = latent_size
        self.src_pad_idx = src_pad_idx
        self.aggregator = aggregator
        self.context_feed_forward = nn.Linear(self.encoder_hidden, self.encoder_hidden)
        self.scalar_mix = None
        if scalar_mix:
            self.scalar_mix = ScalarMix(self.max_layer + 1)
        self.global_scorer = GNMTGlobalScorer(
            alpha=2, beta=None, length_penalty="avg", coverage_penalty=None
        )
        self.definition_encoder = LSTM_Encoder(
            self.embeddings._def,
            self.encoder_hidden,
            self.encoder_num_layers,
            self.dropout_dict.src,
            self.dropout_dict.src,
        )
        self.context_encoder = LSTM_Encoder(
            self.embeddings.src,
            self.encoder_hidden,
            self.encoder_num_layers,
            self.dropout_dict.src,
            self.dropout_dict.src,
        )

        self.decoder = LSTM_Decoder(
            embeddings.tgt,
            hidden=self.decoder_hidden,
            encoder_hidden=self.encoder_hidden,
            num_layers=self.decoder_num_layers,
            word_dropout=word_dropout_p,
            teacher_forcing_p=teacher_forcing_p,
            attention="general" if attentional else None,
            dropout=DotMap({"input": 0.5, "output": 0.5}),
            decoder="VDM" if self.variational else "LSTM",
            variational=self.variational,
            latent_size=self.latent_size,
        )

        self.target_kl = 1.0
        self.definition_feed_forward = nn.Linear(
            self.encoder_hidden, self.encoder_hidden
        )
        self.mean_layer = nn.Linear(self.latent_size, self.latent_size)
        self.logvar_layer = nn.Linear(self.latent_size, self.latent_size)
        self.w_z_post = nn.Sequential(
            nn.Linear(self.encoder_hidden * 2, self.latent_size), nn.Tanh()
        )
        self.mean_prime_layer = nn.Linear(self.latent_size, self.latent_size)
        self.logvar_prime_layer = nn.Linear(self.latent_size, self.latent_size)
        self.w_z_prior = nn.Sequential(
            nn.Linear(self.encoder_hidden, self.latent_size), nn.Tanh()
        )
        self.z_project = nn.Sequential(
            nn.Linear(self.latent_size, self.decoder_hidden), nn.Tanh()
        )

    def forward(
        self,
        input,
        seq_lens,
        span_token_ids,
        target,
        target_lens,
        definition=None,
        definition_lens=None,
        classification_labels=None,
        sentence_mask=None,
    ):
        batch_size, tgt_len = target.shape

        # (batch_size,seq_len,hidden_size), (batch_size,hidden_size), (num_layers,batch_size,seq_len,hidden_size)
        _, last_hidden_layer = self.context_encoder(input, seq_lens, initial_state=None)
        definition_representation, _ = self.definition_encoder(
            definition, definition_lens, initial_state=None
        )

        span_ids = self._id_extractor(tokens=span_token_ids, batch=input, lens=seq_lens)
        span_representation, hidden_states = self._span_aggregator(
            all_hidden_layers if self.scalar_mix is not None else last_hidden_layer,
            sequence_mask(seq_lens),
            span_ids,
        )
        span_representation = self.context_feed_forward(span_representation)

        definition_representation = self.definition_feed_forward(
            definition_representation
        )

        post_project = self.w_z_post(
            torch.cat([span_representation, definition_representation], -1)
        )
        prior_project = self.w_z_prior(span_representation)

        mu = self.mean_layer(post_project)
        logvar = self.logvar_layer(post_project)

        mu_prime = self.mean_prime_layer(prior_project)
        logvar_prime = self.logvar_prime_layer(prior_project)

        z = mu + torch.exp(logvar * 0.5) * torch.randn_like(logvar)
        span_representation = self.z_project(z)
        KLD = kl_divergence(
            Normal(mu, torch.exp(logvar * 0.5)),
            Normal(mu_prime, torch.exp(logvar_prime * 0.5)),
        )
        kl_mask = (KLD > (self.target_kl / self.latent_size)).float()
        fake_loss_kl = (kl_mask * KLD).sum(dim=1)

        predictions, logits = self.decoder(
            target, target_lens, span_representation, hidden_states, seq_lens,
        )

        loss = (
            F.cross_entropy(
                logits.view(batch_size * (tgt_len - 1), -1),
                target[:, 1:].contiguous().view(-1),
                ignore_index=self.embeddings.tgt.padding_idx,
                reduction="none",
            )
            .view(batch_size, tgt_len - 1)
            .sum(1)
        )

        perplexity = F.cross_entropy(
            logits.view(batch_size * (tgt_len - 1), -1),
            target[:, 1:].contiguous().view(-1),
            ignore_index=self.embeddings.tgt.padding_idx,
            reduction="mean",
        ).exp()
        return DotMap(
            {
                "predictions": predictions,
                "logits": logits,
                "loss": loss,
                "perplexity": perplexity,
                "fake_kl": fake_loss_kl,
                "kl": KLD,
                "cosine_loss": cosine_loss,
            }
        )

    def _validate(
        self,
        input,
        seq_lens,
        span_token_ids,
        target,
        target_lens,
        decode_strategy,
        definition=None,
        definition_lens=None,
        sentence_mask=None,
    ):
        batch_size, tgt_len = target.shape

        # (batch_size,seq_len,hidden_size), (batch_size,hidden_size), (num_layers,batch_size,seq_len,hidden_size)
        _, last_hidden_layer = self.context_encoder(input, seq_lens, initial_state=None)
        definition_representation, _ = self.definition_encoder(
            definition, definition_lens, initial_state=None
        )

        span_ids = self._id_extractor(tokens=span_token_ids, batch=input, lens=seq_lens)
        span_representation, hidden_states = self._span_aggregator(
            all_hidden_layers if self.scalar_mix is not None else last_hidden_layer,
            sequence_mask(seq_lens),
            span_ids,
        )
        span_representation = self.context_feed_forward(span_representation)

        definition_representation = self.definition_feed_forward(
            definition_representation
        )

        post_project = self.w_z_post(
            torch.cat([span_representation, definition_representation], -1)
        )
        prior_project = self.w_z_prior(span_representation)

        mu = self.mean_layer(post_project)
        logvar = self.logvar_layer(post_project)

        mu_prime = self.mean_prime_layer(prior_project)
        logvar_prime = self.logvar_prime_layer(prior_project)

        z = mu + torch.exp(logvar * 0.5) * torch.randn_like(logvar)
        span_representation = self.z_project(z)
        KLD = (
            kl_divergence(
                Normal(mu, torch.exp(logvar * 0.5)),
                Normal(mu_prime, torch.exp(logvar_prime * 0.5)),
            )
            .sum(1)
            .mean()
        )

        memory_bank = hidden_states if self.decoder.attention else None
        _, logits = self.decoder(
            target, target_lens, span_representation, memory_bank, seq_lens,
        )

        loss = F.cross_entropy(
            logits.view(batch_size * (tgt_len - 1), -1),
            target[:, 1:].contiguous().view(-1),
            ignore_index=self.embeddings.tgt.padding_idx,
        )

        ppl = loss.exp()
        beam_results = self._strategic_decode(
            target,
            target_lens,
            decode_strategy,
            memory_bank,
            seq_lens,
            span_representation,
        )
        return DotMap(
            {
                "predictions": beam_results["predictions"],
                "logits": logits.view(batch_size * (tgt_len - 1), -1),
                "loss": loss,
                "perplexity": ppl,
                "kl": KLD,
            }
        )

    def _strategic_decode(
        self,
        target,
        target_lens,
        decode_strategy,
        memory_bank,
        memory_lengths,
        span_representation,
    ):
        """Translate a batch of sentences step by step using cache.
        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
        generate translation step by step.
        Returns:
            results (dict): The translation results.
        """

        parallel_paths = decode_strategy.parallel_paths  # beam_size

        # (0) Prep the components of the search.
        # use_src_map = self.copy_attn
        batch_size, max_len = target.shape
        # Initialize the hidden states
        self.decoder.init_state(
            span_representation, encoder_hidden=memory_bank, src_lens=memory_lengths
        )

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            # "gold_score": self._gold_score(
            #    batch, memory_bank, src_lengths, src_vocabs, use_src_map,
            #    enc_states, batch_size, src)
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None  # batch.src_map if use_src_map else None
        fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(
            memory_bank, memory_lengths, src_map, device="cuda"
        )
        if fn_map_state is not None:
            self.decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions

            logits, attn = self.decoder.generate(
                decoder_input, memory_bank, memory_lengths
            )

            decode_strategy.advance(F.log_softmax(logits, 1), attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if memory_bank is not None:
                    if isinstance(memory_bank, tuple):
                        memory_bank = tuple(
                            x.index_select(0, select_indices) for x in memory_bank
                        )
                    else:
                        memory_bank = memory_bank.index_select(0, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                self.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        # if self.decoder.attention:
        #    results["alignment"] = self._align_forward(
        #        batch, decode_strategy.predictions
        #    )
        # else:
        #    results["alignment"] = [[] for _ in range(batch_size)]
        return results

    def _span_aggregator(
        self, hidden_states, input_mask, span_ids, layer_no: int = None,
    ):

        if layer_no is not None:
            hidden_states = hidden_states[layer_no]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[: self.max_layer + 1]
            hidden_states = self.scalar_mix(hidden_states, mask=input_mask)
        if self.aggregator == "span":
            span = self.span_extractor(hidden_states, span_ids).squeeze(
                1
            )  # As we will only be extracting one span per sequence
        elif self.aggregator == "mean":
            extracted_embeddings, span_mask = batched_span_select(
                hidden_states, span_ids
            )
            extracted_embeddings = (
                extracted_embeddings * span_mask.type(torch.float).unsqueeze(-1)
            ).squeeze(1)
            lengths = (span_mask == True).sum(-1) + 1e-20  # To avoid zero division
            span = extracted_embeddings.sum(1) / lengths
        return span, hidden_states
