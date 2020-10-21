import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import (
    InputAttention,
    GRU_Decoder,
    LSTM_Encoder,
    CharCNN,
    BoWLoss,
    LSTMWordAttention,
)
from .skipgram import SkipGramLoss
from .beam import beam_decode
from .beams import beam_search_decode


class Definer(nn.Module):
    def __init__(
        self,
        encoder_hidden: int,
        decoder_hidden: int,
        latent_size: int,
        encoder_num_layers: int,
        decoder_num_layers: int,
        embeddings: dict,
        attn: dict,
        neg_sample_weights: dict,
        neg_sample_no=5,
        conditional=False,
        conditional_classes=15,
        bow_loss_bool=False,
        concat=False,
    ):
        super(Definer, self).__init__()
        # ENCODERS
        self.conditional = conditional
        if conditional:
            self.conditional_size = embeddings["word"][0].embedding_dim

        self.def_encoder = LSTM_Encoder(
            embeddings["def"][0],
            encoder_hidden,
            encoder_num_layers,
            input_dropout=0.5,
            output_dropout=0.5,
        )

        self.lstm_embedding = LSTMWordAttention(embeddings["word"][0], encoder_hidden)
        #        self.attention_embedding = InputAttention(
        #            attn["n_attn_tokens"],
        #            attn["n_attn_embsize"],
        #            attn["n_attn_hid"],
        #            attn["n_attn_dropout"],
        #            embs=embeddings["word"][0],
        #            sparse=False,
        #        )
        self.ff = nn.Sequential(nn.Linear(encoder_hidden * 4, latent_size), nn.Tanh())
        # CONDITIONAL
        if conditional:
            self.conditional_ff = nn.Sequential(
                nn.Linear(encoder_hidden * 2, latent_size), nn.Tanh()
            )
            self.conditional_mean = nn.Linear(latent_size, latent_size)
            self.conditional_logvar = nn.Linear(latent_size, latent_size)
        # VARIATIONAL
        self.mean_linear = nn.Linear(latent_size, latent_size)
        self.logvar_linear = nn.Linear(latent_size, latent_size)
        self.latent2hidden_def = nn.Linear(latent_size, decoder_hidden)

        self.concat = concat
        # DECODERS
        self.def_decoder = GRU_Decoder(
            embeddings["def"][1],
            decoder_hidden,
            decoder_hidden,
            num_layers=1,
            encoder_hidden=encoder_hidden * 2,
            input_dropout=0.5,
            output_dropout=0.5,
            teacher_forcing_p=0.5,
        )
        self.bow_loss_bool = bow_loss_bool
        if bow_loss_bool:
            self.bow_loss = BoWLoss(latent_size, embeddings["def"][1].num_embeddings)

    def forward(
        self,
        word,
        context,
        definition,
        definition_len,
        context_lens,
        definition_mask,
        context_mask,
        index_of_word,
        class_=None,
    ):

        if self.conditional:
            definition_hidden, _ = self.def_encoder(definition, definition_len)
            word_emb, encoder_hidden = self.lstm_embedding(
                word, context, context_lens, index_of_word
            )

            hidden = torch.cat([definition_hidden, word_emb], -1)

            hidden = self.ff(hidden)

            mean_post, logvar_post, z = self.hidden2z(hidden)

            prior_project = self.conditional_ff(word_emb)
            mean_prior = self.conditional_mean(prior_project)
            logvar_prior = self.conditional_logvar(prior_project)

            decoder_hidden = self.latent2hidden_def(z)

            def_loss, def_preds, def_proper_loss = self.def_decoder(
                decoder_hidden, definition, decoder_hidden, context_mask, encoder_hidden
            )

            BoWLoss = 0
            if self.bow_loss_bool:
                BoWLoss = self.bow_loss(z, definition, definition_mask)

            KLD = self.kl_div(
                mean_post,
                logvar_post,
                mu_prior=mean_prior,
                log_sigma_prior=logvar_prior,
            )

            return_dict = {
                "loss": {"definition": def_loss, "skipgram": 0, "bow": BoWLoss,},
                "kld": KLD,
                "preds": {"definition": def_preds,},
                "ppl": torch.exp(def_proper_loss),
            }
            return return_dict

    def kl_div(self, mu_post, log_sigma_post, mu_prior=None, log_sigma_prior=None):
        sigma_post = torch.exp(log_sigma_post)
        sigma_prior = torch.exp(log_sigma_prior)
        if self.conditional:
            KLD = (
                log_sigma_prior
                - log_sigma_post
                + (
                    sigma_post * sigma_post
                    + (mu_post - mu_prior) * (mu_post - mu_prior)
                )
                / (2.0 * sigma_prior * sigma_prior)
                - 0.5
            )
        else:
            KLD = -0.5 * torch.sum(
                1 + log_sigma_post - mu_post.pow(2) - log_sigma_post.exp()
            )

        return KLD

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def hidden2z(self, hidden):
        mean = self.mean_linear(hidden)
        logvar = self.logvar_linear(hidden)
        z = self.reparameterize(mean, logvar)
        return mean, logvar, z

    def idx2hot(self, idx):
        assert torch.max(idx).item() < self.conditional_size

        if idx.dim() == 1:
            idx = idx.unsqueeze(1)

        onehot = torch.zeros(idx.size(0), self.conditional_size)
        onehot.scatter_(1, idx, 1)

        return onehot.cuda()

    def inference(
        self,
        word,
        context,
        definition,
        definition_len,
        context_lens,
        definition_mask,
        context_mask,
        index_of_word,
        vocab,
        max_len,
        beam_size,
        class_=None,
    ):
        self.eval()
        if self.conditional:
            word_emb, encoder_hidden = self.lstm_embedding(
                word, context, context_lens, index_of_word
            )
            prior_project = self.conditional_ff(word_emb)
            z = self.conditional_mean(prior_project)

            def_decoder_hidden = self.latent2hidden_def(z)
            beam_preds, _ = beam_search_decode(
                self,
                vocab,
                max_len,
                beam_size,
                context_mask,
                encoder_hidden,
                def_decoder_hidden,
            )
            def_loss, def_preds, def_proper_loss = self.def_decoder(
                def_decoder_hidden,
                definition,
                def_decoder_hidden,
                context_mask,
                encoder_hidden,
            )
            BoWLoss = 0
            if self.bow_loss_bool:
                BoWLoss = self.bow_loss(z, definition, definition_mask)

            return_dict = {
                "loss": {"definition": def_loss, "bow": BoWLoss,},
                "kld": torch.ones(z.shape),
                "preds": {"definition": beam_preds,},
                "ppl": torch.exp(def_proper_loss),
            }
            return return_dict


# class PronouncerDefiner(nn.Module):
#    def __init__(
#        self,
#        encoder_hidden: int,
#        decoder_hidden: int,
#        latent_size: int,
#        encoder_num_layers: int,
#        decoder_num_layers: int,
#        embeddings: dict,
#        ch: dict,
#        attn: dict,
#        neg_sample_weights: dict,
#        neg_sample_no=5,
#        conditional=False,
#        conditional_classes=15,
#    ):
#        super(PronouncerDefiner, self).__init__()
#        # ENCODERS
#        self.conditional = conditional
#        if conditional:
#            self.conditional_size = embeddings["word"][0].embedding_dim
#        self.char_encoder = CharCNN(
#            ch["n_ch_tokens"],
#            ch["ch_maxlen"],
#            ch["ch_emb_size"],
#            ch["ch_feature_maps"],
#            ch["ch_kernel_sizes"],
#            embs=embeddings["char"][0],
#        )
#        self.def_encoder = LSTM_Encoder(
#            embeddings["def"][0],
#            encoder_hidden,
#            encoder_num_layers,
#            input_dropout=0.5,
#            output_dropout=0.5,
#        )
#        if encoder == 'ia':
#            self.attention_embedding = InputAttention(
#                attn["n_attn_tokens"],
#                attn["n_attn_embsize"],
#                attn["n_attn_hid"],
#                attn["n_attn_dropout"],
#                embs=embeddings["word"][0],
#                sparse=False,
#            )
#        elif encoder == 'lstmia':
#
#        self.ff = nn.Linear(
#            sum([encoder_hidden * 2, attn["n_attn_embsize"], 128]),
#            sum([encoder_hidden * 2, attn["n_attn_embsize"], 128]),
#        )
#        # CONDITIONAL
#        if conditional:
#            self.conditional_mean = nn.Linear(
#                embeddings["word"][0].embedding_dim, latent_size
#            )
#            self.conditional_logvar = nn.Linear(
#                embeddings["word"][0].embedding_dim, latent_size
#            )
#        # VARIATIONAL
#        self.mean_linear = nn.Linear(
#            sum([encoder_hidden * 2, attn["n_attn_embsize"], 128]), latent_size
#        )
#        self.logvar_linear = nn.Linear(
#            sum([encoder_hidden * 2, attn["n_attn_embsize"], 128]), latent_size
#        )
#        self.latent2hidden_def = nn.Linear(latent_size, decoder_hidden)
#        self.latent2hidden_char = nn.Linear(latent_size, decoder_hidden)
#
#        # DECODERS
#        self.def_decoder = GRU_Decoder(
#            embeddings["def"][1],
#            decoder_hidden,
#            decoder_hidden,
#            num_layers=1,
#            input_dropout=0.5,
#            output_dropout=0.5,
#            teacher_forcing_p=0.5,
#        )
#        self.char_decoder = GRU_Decoder(
#            embeddings["char"][1],
#            decoder_hidden,
#            decoder_hidden,
#            num_layers=1,
#            input_dropout=0.5,
#            output_dropout=0.5,
#            teacher_forcing_p=0.5,
#        )
#
#        # SKIPGRAM
#        self.skipgram = SkipGramLoss(
#            latent_size=latent_size,
#            n_attn_tokens=attn["n_attn_tokens"],
#            n_attn_embsize=attn["n_attn_embsize"],
#            n_attn_hid=attn["n_attn_hid"],
#            attn_dropout=attn["n_attn_dropout"],
#            neg_sample_no=neg_sample_no,
#            neg_sample_weights=neg_sample_weights,
#            emb1_lookup=embeddings["word"][1],
#            sparse=False,
#        )
#        self.bow_loss = BoWLoss(latent_size, embeddings["def"][1].num_embeddings)
#
#    def forward(
#        self,
#        word,
#        context,
#        pronounciation,
#        definition,
#        definition_len,
#        context_lens,
#        definition_mask,
#        class_=None,
#    ):
#
#        if self.conditional:
#            pronounciation_hidden = self.char_encoder(pronounciation)
#            definition_hidden, _ = self.def_encoder(definition, definition_len)
#            word_emb = self.attention_embedding(word, context)
#
#            hidden = torch.cat([pronounciation_hidden, definition_hidden, word_emb], -1)
#
#            hidden = self.ff(hidden)
#            mean_post, logvar_post, z = self.hidden2z(hidden)
#
#            mean_prior = self.conditional_mean(word_emb)
#            logvar_prior = self.conditional_logvar(word_emb)
#
#            decoder_hidden = self.latent2hidden_def(z)
#            char_decoder_hidden = self.latent2hidden_char(z)
#
#            def_loss, def_preds = self.def_decoder(
#                decoder_hidden, definition, decoder_hidden
#            )
#
#            pronounciation_loss, pronounciation_preds = self.char_decoder(
#                char_decoder_hidden, pronounciation, char_decoder_hidden
#            )
#
#            skipgram_loss = self.skipgram(z, context, context_lens)
#
#            BoWLoss = self.bow_loss(z, definition, definition_mask)
#            KLD = self.kl_div(
#                mean_post,
#                logvar_post,
#                mu_prior=mean_prior,
#                log_sigma_prior=logvar_prior,
#            )
#            return_dict = {
#                "loss": {
#                    "pronounciation": pronounciation_loss,
#                    "definition": def_loss,
#                    "skipgram": skipgram_loss,
#                    "bow": BoWLoss,
#                },
#                "kld": KLD,
#                "preds": {
#                    "definition": def_preds,
#                    "pronounciation": pronounciation_preds,
#                },
#            }
#            return return_dict
#        else:
#            pronounciation_hidden = self.char_encoder(pronounciation)
#            definition_hidden, _ = self.def_encoder(definition, definition_len)
#            word_emb = self.attention_embedding(word, context)
#
#            hidden = torch.cat([pronounciation_hidden, definition_hidden, word_emb], -1)
#
#            hidden = self.ff(hidden)
#            mean, logvar, z = self.hidden2z(hidden)
#
#            decoder_hidden = self.latent2hidden_def(z)
#            char_decoder_hidden = self.latent2hidden_char(z)
#
#            def_loss, def_preds = self.def_decoder(
#                decoder_hidden, definition, decoder_hidden
#            )
#
#            pronounciation_loss, pronounciation_preds = self.char_decoder(
#                char_decoder_hidden, pronounciation, char_decoder_hidden
#            )
#
#            skipgram_loss = self.skipgram(z, context, context_lens)
#
#            BoWLoss = self.bow_loss(z, definition, definition_mask)
#            KLD = self.kl_div(mean, logvar)
#            return_dict = {
#                "loss": {
#                    "pronounciation": pronounciation_loss,
#                    "definition": def_loss,
#                    "skipgram": skipgram_loss,
#                    "bow": BoWLoss,
#                },
#                "kld": KLD,
#                "preds": {
#                    "definition": def_preds,
#                    "pronounciation": pronounciation_preds,
#                },
#                "variational": {"mean": mean, "logvar": logvar},
#            }
#            return return_dict
#
#    def kl_div(self, mu_post, log_sigma_post, mu_prior=None, log_sigma_prior=None):
#        sigma_post = torch.exp(log_sigma_post)
#        sigma_prior = torch.exp(log_sigma_prior)
#        if self.conditional:
#            KLD = (
#                log_sigma_prior
#                - log_sigma_post
#                + (
#                    sigma_post * sigma_post
#                    + (mu_post - mu_prior) * (mu_post - mu_prior)
#                )
#                / (2.0 * sigma_prior * sigma_prior)
#                - 0.5
#            )
#        else:
#            KLD = -0.5 * torch.sum(
#                1 + log_sigma_post - mu_post.pow(2) - log_sigma_post.exp()
#            )
#
#        return KLD
#
#    def reparameterize(self, mu, logvar):
#        std = torch.exp(0.5 * logvar)
#        eps = torch.randn_like(std)
#        return mu + eps * std
#
#    def hidden2z(self, hidden):
#        mean = self.mean_linear(hidden)
#        logvar = self.logvar_linear(hidden)
#        z = self.reparameterize(mean, logvar)
#        return mean, logvar, z
#
#    def idx2hot(self, idx):
#        assert torch.max(idx).item() < self.conditional_size
#
#        if idx.dim() == 1:
#            idx = idx.unsqueeze(1)
#
#        onehot = torch.zeros(idx.size(0), self.conditional_size)
#        onehot.scatter_(1, idx, 1)
#
#        return onehot.cuda()
#
#    def inference(
#        self,
#        word,
#        context,
#        pronounciation,
#        definition,
#        definition_len,
#        context_lens,
#        definition_mask,
#        class_=None,
#    ):
#        self.eval()
#        if self.conditional:
#            word_emb = self.attention_embedding(word, context)
#            mu = self.conditional_mean(word_emb)
#            logvar = self.conditional_logvar(word_emb)
#            z = self.reparameterize(mu, logvar)
#
#            def_decoder_hidden = self.latent2hidden_def(z)
#            char_decoder_hidden = self.latent2hidden_char(z)
#
#            def_loss, def_preds = self.def_decoder(
#                def_decoder_hidden, definition, def_decoder_hidden
#            )
#
#            pronounciation_loss, pronounciation_preds = self.char_decoder(
#                char_decoder_hidden, pronounciation, char_decoder_hidden
#            )
#
#            skipgram_loss = self.skipgram(z, context, context_lens)
#
#            BoWLoss = self.bow_loss(z, definition, definition_mask)
#
#            return_dict = {
#                "loss": {
#                    "pronounciation": pronounciation_loss,
#                    "definition": def_loss,
#                    "skipgram": skipgram_loss,
#                    "bow": BoWLoss,
#                },
#                "kld": torch.empty(z.shape),
#                "preds": {
#                    "definition": def_preds,
#                    "pronounciation": pronounciation_preds,
#                },
#                "variational": {"mean": mu, "logvar": logvar},
#            }
#            return return_dict
#        else:
#            pronounciation_hidden = self.char_encoder(pronounciation)
#            definition_hidden, _ = self.def_encoder(definition, definition_len)
#            word_emb = self.attention_embedding(word, context)
#
#            hidden = torch.cat([pronounciation_hidden, definition_hidden, word_emb], -1)
#
#            hidden = self.ff(hidden)
#            mean, logvar, z = self.hidden2z(hidden)
#
#            decoder_hidden = self.latent2hidden_def(z)
#            char_decoder_hidden = self.latent2hidden_char(z)
#
#            def_loss, def_preds = self.def_decoder(
#                decoder_hidden, definition, decoder_hidden
#            )
#
#            pronounciation_loss, pronounciation_preds = self.char_decoder(
#                char_decoder_hidden, pronounciation, char_decoder_hidden
#            )
#
#            skipgram_loss = self.skipgram(z, context, context_lens)
#
#            BoWLoss = self.bow_loss(z, definition, definition_mask)
#            KLD = self.kl_div(mean, logvar)
#            return_dict = {
#                "loss": {
#                    "pronounciation": pronounciation_loss,
#                    "definition": def_loss,
#                    "skipgram": skipgram_loss,
#                    "bow": BoWLoss,
#                },
#                "kld": KLD,
#                "preds": {
#                    "definition": def_preds,
#                    "pronounciation": pronounciation_preds,
#                },
#                "variational": {"mean": mean, "logvar": logvar},
#            }
#            return return_dict
