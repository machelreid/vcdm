from config import StrictConfigParser
from trainer import build_trainer
from model import DefinitionProbing
from data import get_dm_conf, DataMaker
from modules import get_pretrained_transformer
import os
import torch
import torch.nn as nn
from dotmap import DotMap
import hashlib
import json

from embeddings import Word2Vec

config_parser = StrictConfigParser(default=os.path.join("config", "config.yaml"))

if __name__ == "__main__":

    config = config_parser.parse_args()

    use_cuda = config.device == "cuda" and torch.cuda.is_available()

    torch.manual_seed(config.seed)

    example_field = get_dm_conf(config.encoder, "example")
    word_field = get_dm_conf(config.encoder, "word")
    definition_field = get_dm_conf("normal", "definition")

    data_fields = [example_field, word_field, definition_field]
    if config.variational or config.defbert:
        definition_ae_field = get_dm_conf(config.encoder, "definition_ae")
        data_fields.append(definition_ae_field)

    device = torch.device("cuda" if use_cuda else "cpu")

    config.update(
        {
            "serialization_dir": config.serialization_dir
            + config.dataset
            + "/"
            + hashlib.sha224(
                json.dumps(dict(config.to_dict()), sort_keys=True).encode()
            ).hexdigest()[:6]
        }
    )
    ############### DATA ###############
    datamaker = DataMaker(data_fields, config.datapath)
    datamaker.build_data(
        config.dataset,
        max_len=config.max_length,
        lowercase=config.lowercase,
        shared_vocab_fields=["example", "word"],
    )
    ####################################
    ####################################

    ############### MODEL ##############
    embeddings = DotMap(
        {
            "tgt": nn.Embedding.from_pretrained(
                Word2Vec(datamaker.vocab.definition.itos),
                freeze=False,
                padding_idx=datamaker.vocab.definition.stoi["<pad>"],
            )
        }
    )
    embeddings.tgt.unk_idx, embeddings.tgt.padding_idx = (
        datamaker.vocab.definition.stoi["<unk>"],
        datamaker.vocab.definition.stoi["<pad>"],
    )

    dropout = DotMap(
        {
            "src": {
                "input": config.src_input_dropout,
                "output": config.src_output_dropout,
            },
            "tgt": {
                "input": config.tgt_input_dropout,
                "output": config.tgt_output_dropout,
            },
            "tgt_word_dropout": config.tgt_word_dropout,
            "src_word_dropout": config.src_word_dropout,
        }
    )

    encoder = get_pretrained_transformer(config.encoder)
    if config.variational or config.defbert:
        if config.tied:
            definition_encoder = encoder
        else:
            definition_encoder = get_pretrained_transformer(config.encoder)
    else:
        definition_encoder = None

    model = DefinitionProbing(
        encoder=encoder,
        encoder_pretrained=True,
        encoder_frozen=config.encoder_frozen,
        decoder_hidden=config.decoder_hidden,
        embeddings=embeddings,
        max_layer=config.max_layer,
        src_pad_idx=datamaker.vocab.example.pad_token_id,
        teacher_forcing_p=config.teacher_forcing_p,
        attentional=config.attentional,
        aggregator=config.aggregator,
        variational=config.variational,
        latent_size=config.latent_size,
        word_dropout_p=config.tgt_word_dropout,
        definition_encoder=definition_encoder,
        decoder_num_layers=config.decoder_num_layers,
    ).to(config.device)

    ####################################
    ####################################

    ########## TRAINING LOOP ###########
    trainer = build_trainer(model, config, datamaker)
    with open(config.serialization_dir + "/config.json", "w") as f:
        json.dump(dict(config.to_dict()), f)
    with open(config.serialization_dir + "/model_architecture", "w") as f:
        f.write(
            repr(model)
            + "\nParameter Count:"
            f" {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

    try:
        for i in range(config.max_epochs):
            train_out = trainer._train(config.train_batch_size)
            if train_out is None:
                break
            valid_out = trainer._validate(config.valid_batch_size)
        test_out = trainer._test(config.valid_batch_size)
    except KeyboardInterrupt:
        print("Stopping training, train counter =", trainer._train_counter)
