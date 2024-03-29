import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class StrawmanNet(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )

    # pylint: disable=arguments-differ
    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def _step(self, batch, *args, **kwargs):
        x, y = batch["features"], batch["label"]
        z = self(x)
        return F.cross_entropy(z, y)

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, val_batch, batch_id):
        loss = self._step(val_batch)
        self.log("val_loss", loss)
        return loss


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


# https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(pl.LightningModule):
    r"""Inject some information about the relative or absolute position of the
        tokens in the sequence. The positional encodings have the same dimension
        as the embeddings, so that the two can be summed. Here, we use sine and
        cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[pos]
        return self.dropout(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class TransformerModel(pl.LightningModule):
    """Container module with a positional encoder."""

    def __init__(
        self,
        d_input,
        d_model,
        max_len=1024,
        dropout=0.1,
        lr=1e-3,
        warmup=500,
        max_iters=20000,
        predict_reverse=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # add some extra-nonlinearity for embedding the input data
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.d_input, 1024),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(1024, self.hparams.d_model),
            nn.ReLU(),
        )
        self.input_net.apply(init_weights)

        self.pos_encoder = PositionalEncoding(
            self.hparams.d_model,
            dropout=self.hparams.dropout,
            max_len=self.hparams.max_len,
        )
        self.transformer = nn.Transformer(
            self.hparams.d_model, dropout=self.hparams.dropout, **kwargs
        )

    def _generate_square_subsequent_mask(self, sz):
        """Create a mask that masks starting from the right/

        >>> (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        tensor([[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]])
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 0).transpose(0, 1)
        # mask = (
        #     mask.float()
        #     .masked_fill(mask == 0, float("-inf"))
        #     .masked_fill(mask == 1, float(0.0))
        # )
        return mask

    def _create_subsequence_mask(self, src, tgt):
        """Create the subsequence masks for src and target."""
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        # if we're predicting the reverse sequence, then we don't need any sort
        # of masking aside from the ones that we get. If anything, having a mask
        # in the wrong direction may cause issues.
        # if self.hparams.predict_reverse:
        if False:
            tgt_mask = (
                torch.zeros((tgt_seq_len, tgt_seq_len)).type(torch.bool).to(self.device)
            )
        else:
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(
                self.device
            )
        src_mask = (
            torch.zeros((src_seq_len, src_seq_len)).type(torch.bool).to(self.device)
        )
        return src_mask, tgt_mask

    def forward(self, src, tgt, src_pos, tgt_pos, **kwargs):
        src_mask, tgt_mask = self._create_subsequence_mask(src, tgt)
        # NOTE: we also pass in the padding mask through here
        z = self.transformer(
            self.pos_encoder(src, src_pos),
            self.pos_encoder(tgt, tgt_pos),
            src_mask,
            tgt_mask,
            **kwargs
        )
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup=self.hparams.warmup,
            max_iters=self.hparams.max_iters,
        )
        return [optimizer], [lr_scheduler]

    def _step(self, batch, *args, **kwargs):
        x, y, src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos = (
            batch["src"],
            batch["tgt"],
            batch["src_key_padding_mask"],
            batch["tgt_key_padding_mask"],
            batch["src_pos"],
            batch["tgt_pos"],
        )
        # reshape x and y to be [batch_size, seq_len, embed_dim] and reorder
        # dimensions to be [seq_len, batch_size, embed_dim]
        x = self.input_net(x.view(x.shape[0], -1, self.hparams.d_input))
        y = self.input_net(y.view(y.shape[0], -1, self.hparams.d_input))

        x = x.transpose(0, 1)
        y = y.transpose(0, 1)

        z = self(
            x,
            y,
            src_pos.transpose(0, 1),
            tgt_pos.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask.type(torch.bool),
            tgt_key_padding_mask=tgt_key_padding_mask.type(torch.bool),
        )
        mask = (tgt_key_padding_mask == 0).transpose(0, 1)
        # NOTE: what is the best loss to use here? Does it even make sense to
        # use the cross entropy loss?
        # return F.cosine_embedding_loss(
        #     z[mask], y[mask], torch.ones(z.shape[0] * z.shape[1]).to(self.device)
        # )
        return F.mse_loss(z[mask], y[mask])

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, val_batch, batch_id):
        loss = self._step(val_batch)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_id):
        x, src_key_padding_mask, src_pos = (
            batch["src"],
            batch["src_key_padding_mask"],
            batch["src_pos"],
        )
        x = self.input_net(x.view(x.shape[0], -1, self.hparams.d_input))
        x = x.transpose(0, 1)
        # this is a bit ugly, could be cleaned up a bit to match the _step function
        z = self.transformer.encoder(
            self.pos_encoder(x, src_pos.transpose(0, 1)),
            src_key_padding_mask=src_key_padding_mask.type(torch.bool),
        )
        return {
            "customer_index": batch["customer_index"],
            "prediction": z.transpose(0, 1).reshape(z.shape[1], -1),
        }


class TransformerEmbeddingModel(pl.LightningModule):
    """Container module with a positional encoder."""

    def __init__(
        self,
        d_input,
        d_model,
        d_embed,
        seq_len,
        num_layers=6,
        nhead=8,
        max_len=1024,
        dropout=0.1,
        lr=1e-3,
        warmup=500,
        max_iters=20000,
        predict_reverse=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # add some extra-nonlinearity for embedding the input data

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.d_input, self.hparams.d_model),
        )

        self.pos_encoder = PositionalEncoding(
            self.hparams.d_model,
            dropout=self.hparams.dropout,
            max_len=self.hparams.max_len,
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.hparams.d_model,
                self.hparams.nhead,
                dropout=self.hparams.dropout,
                **kwargs
            ),
            self.hparams.num_layers,
        )

        # layer used for embedding the input data so it maps to the output data
        dim = self.hparams.d_model * self.hparams.seq_len
        self.output_net = nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            nn.Linear(dim, self.hparams.d_embed),
        )

        # used to map the target to the data embedding
        self.predict_net = nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.d_model, self.hparams.d_embed),
        )

        self.input_net.apply(init_weights)
        self.output_net.apply(init_weights)
        self.predict_net.apply(init_weights)

    def forward(self, src, src_pos, src_key_padding_mask):
        z = self.input_net(src.view(src.shape[0], -1, self.hparams.d_input))
        sz = z.shape[1]
        # src_mask = torch.zeros((sz, sz), device=self.device).type(torch.bool)
        src_mask = (
            (torch.triu(torch.ones(sz, sz, device=self.device)) == 0)
            .flip(0)
            .type(torch.bool)
        )
        # reshape x and y to be [batch_size, seq_len, embed_dim] and reorder
        # dimensions to be [seq_len, batch_size, embed_dim]
        z = self.transformer(
            self.pos_encoder(z.transpose(0, 1), src_pos.transpose(0, 1)),
            src_mask,
            src_key_padding_mask=src_key_padding_mask.type(torch.bool),
        )
        # [batch_size, embed_dim]
        z = self.output_net(z.transpose(0, 1).reshape(src.shape[0], -1))
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup=self.hparams.warmup,
            max_iters=self.hparams.max_iters,
        )
        return [optimizer], [lr_scheduler]

    def _step(self, batch, *args, **kwargs):
        x, y, src_key_padding_mask, _, src_pos, tgt_pos = (
            batch["src"],
            batch["tgt"],
            batch["src_key_padding_mask"],
            batch["tgt_key_padding_mask"],
            batch["src_pos"],
            batch["tgt_pos"],
        )
        z = self.forward(x, src_pos, src_key_padding_mask)

        y = y.view(y.shape[0], -1, self.hparams.d_input)
        y = self.input_net(y[:, :1, :])
        y = self.pos_encoder(
            y.transpose(0, 1),
            tgt_pos.transpose(0, 1)[:1, :],
        )
        y = self.predict_net(y[0])
        # mask = torch.ones(y.shape[0]).to(self.device)
        return F.mse_loss(z, y)

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, src_key_padding_mask, src_pos = (
            batch["src"],
            batch["src_key_padding_mask"],
            batch["src_pos"],
        )
        return {
            "customer_index": batch["customer_index"],
            "prediction": self.forward(x, src_pos, src_key_padding_mask),
        }
