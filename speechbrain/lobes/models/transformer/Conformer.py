"""Conformer implementation.

Authors
* Jianyuan Zhong 2020
* Samuele Cornell 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import speechbrain as sb
import math
import warnings


from speechbrain.nnet.attention import (
    RelPosMHAXL,
    MultiheadAttention,
    PositionalwiseFeedForward,
)
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.nnet.activations import Swish


class ConvolutionModule(nn.Module):
    """This is an implementation of convolution module in Conformer.

    Arguments
    ----------
    input_size : int
        The expected size of the input embedding dimension.
    kernel_size: int, optional
        Kernel size of non-bottleneck convolutional layer.
    bias: bool, optional
        Whether to use bias in the non-bottleneck conv layer.
    activation: torch.nn.Module
         Activation function used after non-bottleneck conv layer.
    dropout: float, optional
         Dropout rate.
    causal: bool, optional
         Whether the convolution should be causal or not.
    dilation: int, optional
         Dilation factor for the non bottleneck conv layer.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = ConvolutionModule(512, 3)
    >>> output = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        input_size,
        kernel_size=31,
        bias=True,
        activation=Swish,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.causal = causal

        if self.causal:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1)
        else:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1) // 2

        self.layer_norm = nn.LayerNorm(input_size)
        self.bottleneck = nn.Sequential(
            # pointwise
            nn.Conv1d(
                input_size, 2 * input_size, kernel_size=1, stride=1, bias=bias
            ),
            nn.GLU(dim=1),
        )
        # depthwise
        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
            groups=input_size,
            bias=bias,
        )

        self.after_conv = nn.Sequential(
            nn.LayerNorm(input_size),
            activation(),
            # pointwise
            nn.Linear(input_size, input_size, bias=bias),
            nn.Dropout(dropout),
        )

    def _do_conv(self, x, inhibit_padding: bool):
        out = self.layer_norm(x)
        out = out.transpose(1, 2)
        out = self.bottleneck(out)

        if not inhibit_padding:
            out = self.conv(out)
        else:
            # let's keep backwards compat by pointing at the weights from the
            # already declared Conv1d.

            # we do not need to edit bottleneck as it is pointwise (i.e. time
            # step by time step), thus, it doesn't need padding along the
            # time dimension
            out = F.conv1d(
                out,
                weight=self.conv.weight,
                bias=self.conv.bias,
                stride=self.conv.stride,
                padding=0,
                dilation=self.conv.dilation,
                groups=out.shape[-2],
            )

        if self.causal:
            # chomp
            out = out[..., : -self.padding]
        out = out.transpose(1, 2)
        out = self.after_conv(out)
        return out

    def forward(self, x, mask=None, chunk_size=None):
        """ Processes the input tensor x and returns the output an output tensor"""

        # ref: Dynamic chunk convolution for unified streaming and non-streaming
        # conformer ASR
        # https://www.amazon.science/publications/dynamic-chunk-convolution-for-unified-streaming-and-non-streaming-conformer-asr
        # split the input into chunks of size `chunk_size`, but for each chunk
        # provide a left context for left chunk dependencies to be possible.
        if chunk_size is not None:
            # chances are chunking+causal is unintended; i don't know where it
            # may make sense, but if it does to you, feel free to implement it.
            assert (
                not self.causal
            ), "Chunked convolution not supported with causal padding"

            chunk_left_context = self.padding
            chunk_count = int(math.ceil(x.shape[1] / chunk_size))

            applied_left_context = [
                min(
                    chunk_left_context,
                    i * chunk_size,
                )
                for i in range(chunk_count)
            ]

            # add left context
            # the left context effectively becomes "left padding", we do not
            # want to keep any convolution results centered on the left context
            out = [
                x[:,i * chunk_size - applied_left_context[i]:(i + 1) * chunk_size,...]
                for i in range(chunk_count)
            ]

            # add missing left padding (if we don't have enough context)
            # add right padding (as we disable default padding, which is
            # forcefully bidirectional)
            out = [
                F.pad(out[i], (
                    0,
                    0,
                    chunk_left_context - applied_left_context[i],  # left
                    self.padding  # right
                ))
                for i in range(len(out))
            ]

            # usual, but chunk-wise processing
            out = [
                self._do_conv(chunk, inhibit_padding=True) for chunk in out
            ]

            out = torch.cat(out, 1)
        else:
            out = self._do_conv(x, inhibit_padding=False)

        if mask is not None:
            out.masked_fill_(mask, 0.0)

        return out


class ConformerEncoderLayer(nn.Module):
    """This is an implementation of Conformer encoder layer.

    Arguments
    ----------
    d_model : int
        The expected size of the input embedding.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: torch.nn.Module
         Activation function used in each Conformer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_embs = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoderLayer(d_ffn=512, nhead=8, d_model=512, kernel_size=3)
    >>> output = net(x, pos_embs=pos_embs)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )
        elif attention_type == "RelPosMHAXL":
            # transformerXL style positional encoding
            self.mha_layer = RelPosMHAXL(
                num_heads=nhead,
                embed_dim=d_model,
                dropout=dropout,
                mask_pos_future=causal,
            )

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: torch.Tensor = None,
        chunk_size: Optional[int] = None,
        left_context_chunks: int = -1,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            The mask for the src sequence.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch.
        pos_embs: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the input sequence positional embeddings
        chunk_size: int, optional
            Whether to preform convolution chunking to hide future context,
            useful for chunked conformers in a dynamic chunk training setting
        """
        # TODO: cite paper for chunk size
        # TODO: document left frames

        def yolo_detect_nan(v, txt):
            if v.isnan().count_nonzero().item() > 0:
                print(f"found nan in {txt}: {v}")

        conv_mask: Optional[torch.Tensor] = None
        if src_key_padding_mask is not None:
            conv_mask = src_key_padding_mask.unsqueeze(-1)
        # ffn module
        yolo_detect_nan(x, "pre ffn")
        x = x + 0.5 * self.ffn_module1(x)
        # muti-head attention module
        skip = x
        yolo_detect_nan(x, "pre norm")
        x = self.norm1(x)

        yolo_detect_nan(x, "pre mha")
        x, self_attn = self.mha_layer(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )
        yolo_detect_nan(x, "pre skip")
        x = x + skip
        # convolution module
        yolo_detect_nan(x, "pre conv")
        x = x + self.convolution_module(x, conv_mask, chunk_size=chunk_size)
        # ffn module
        yolo_detect_nan(x, "pre norm2")
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn


class ConformerEncoder(nn.Module):
    """This class implements the Conformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Embedding dimension size.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: torch.nn.Module
         Activation function used in each Confomer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.


    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_emb = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoder(1, 512, 512, 8)
    >>> output, _ = net(x, pos_embs=pos_emb)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        nhead,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    attention_type=attention_type,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)
        self.attention_type = attention_type

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        left_context_chunks: int = -1,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            The mask for the src sequence.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch.
        pos_embs: torch.Tensor, torch.nn.Module,
            Module or tensor containing the input sequence positional embeddings
            If custom pos_embs are given it needs to have the shape (1, 2*S-1, E)
            where S is the sequence length, and E is the embedding dimension.
        chunk_size: int, optional
            Whether to preform convolution chunking to hide future context,
            useful for chunked conformers in a dynamic chunk training setting
        """
        def yolo_detect_nan(v, txt):
            if v.isnan().count_nonzero().item() > 0:
                print(f"found nan in {txt}: {v}")

        if self.attention_type == "RelPosMHAXL":
            if pos_embs is None:
                raise ValueError(
                    "The chosen attention type for the Conformer is RelPosMHAXL. For this attention type, the positional embeddings are mandatory"
                )

        yolo_detect_nan(src, "src in whole enc")
        output = src
        attention_lst = []
        for i, enc_layer in enumerate(self.layers):
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
                chunk_size=chunk_size,
                left_context_chunks=left_context_chunks,
            )
            yolo_detect_nan(output, f"post layer # {i}")
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst


class ConformerDecoderLayer(nn.Module):
    """This is an implementation of Conformer encoder layer.

    Arguments
    ----------
    d_model : int
        The expected size of the input embedding.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: torch.nn.Module, optional
         Activation function used in each Conformer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_embs = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoderLayer(d_ffn=512, nhead=8, d_model=512, kernel_size=3)
    >>> output = net(x, pos_embs=pos_embs)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=True,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if not causal:
            warnings.warn(
                "Decoder is not causal, in most applications it should be causal, you have been warned !"
            )

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )
        elif attention_type == "RelPosMHAXL":
            # transformerXL style positional encoding
            self.mha_layer = RelPosMHAXL(
                num_heads=nhead,
                embed_dim=d_model,
                dropout=dropout,
                mask_pos_future=causal,
            )

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
            tgt: torch.Tensor
                The sequence to the decoder layer.
            memory: torch.Tensor
                The sequence from the last layer of the encoder.
            tgt_mask: torch.Tensor, optional, optional
                The mask for the tgt sequence.
            memory_mask: torch.Tensor, optional
                The mask for the memory sequence.
            tgt_key_padding_mask : torch.Tensor, optional
                The mask for the tgt keys per batch.
            memory_key_padding_mask : torch.Tensor, optional
                The mask for the memory keys per batch.
            pos_emb_tgt: torch.Tensor, torch.nn.Module, optional
                Module or tensor containing the target sequence positional embeddings for each attention layer.
            pos_embs_src: torch.Tensor, torch.nn.Module, optional
                Module or tensor containing the source sequence positional embeddings for each attention layer.
        """
        # ffn module
        tgt = tgt + 0.5 * self.ffn_module1(tgt)
        # muti-head attention module
        skip = tgt
        x = self.norm1(tgt)
        x, self_attn = self.mha_layer(
            x,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            pos_embs=pos_embs_src,
        )
        x = x + skip
        # convolution module
        x = x + self.convolution_module(x)
        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn, self_attn


class ConformerDecoder(nn.Module):
    """This class implements the Transformer decoder.

    Arguments
    ----------
    num_layers: int
        Number of layers.
    nhead: int
        Number of attention heads.
    d_ffn: int
        Hidden size of self-attention Feed Forward layer.
    d_model: int
        Embedding dimension size.
    kdim: int, optional
        Dimension for key.
    vdim: int, optional
        Dimension for value.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
         Activation function used after non-bottleneck conv layer.
    kernel_size : int, optional
        Kernel size of convolutional layer.
    bias : bool, optional
        Whether  convolution module.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.


    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = ConformerDecoder(1, 8, 1024, 512, attention_type="regularMHA")
    >>> output, _, _ = net(tgt, src)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=Swish,
        kernel_size=3,
        bias=True,
        causal=True,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                ConformerDecoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
        tgt: torch.Tensor
            The sequence to the decoder layer.
        memory: torch.Tensor
            The sequence from the last layer of the encoder.
        tgt_mask: torch.Tensor, optional, optional
            The mask for the tgt sequence.
        memory_mask: torch.Tensor, optional
            The mask for the memory sequence.
        tgt_key_padding_mask : torch.Tensor, optional
            The mask for the tgt keys per batch.
        memory_key_padding_mask : torch.Tensor, optional
            The mask for the memory keys per batch.
        pos_emb_tgt: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the target sequence positional embeddings for each attention layer.
        pos_embs_src: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the source sequence positional embeddings for each attention layer.

        """
        output = tgt
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output)

        return output, self_attns, multihead_attns
