import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TransformerModel(pl.LightningModule):
    """Container module with a positional encoder."""

    def __init__(self, d_model, max_len=1024, dropout=0.1, **kwargs):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.transformer = nn.Transformer(d_model, dropout=dropout, **kwargs)

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

        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(self.device)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
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
        x = x.view(x.shape[0], -1, self.d_model).transpose(0, 1)
        y = y.view(y.shape[0], -1, self.d_model).transpose(0, 1)

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
        return F.mse_loss(z[mask], y[mask])

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, val_batch, batch_id):
        loss = self._step(val_batch)
        self.log("val_loss", loss)
        return loss
