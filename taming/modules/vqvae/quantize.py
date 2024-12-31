from __future__ import annotations

import math
from collections import namedtuple
from contextlib import nullcontext
from functools import cache, partial, wraps
from math import ceil, log2
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from torch import Tensor, einsum, int32, nn
from torch.amp import autocast
from torch.distributed import nn as dist_nn
from torch.nn import Module

LossBreakdown = namedtuple(
    "LossBreakdown", ["per_sample_entropy", "codebook_entropy", "commitment", "avg_probs"]
)


class SimVQ(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e,
        e_dim,
        beta=0.25,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=True,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        quant_codebook = self.embedding_proj(self.embedding.weight)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(quant_codebook**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(quant_codebook, "n d -> d n"))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            commit_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3]
            )

        return (z_q, torch.tensor(0.0), min_encoding_indices), LossBreakdown(
            torch.tensor(0.0), torch.tensor(0.0), commit_loss, torch.tensor(0.0)
        )

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class SimVQ1D(SimVQ):
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, "b c h -> b h c").contiguous()
        assert z.shape[-1] == self.e_dim

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        quant_codebook = self.embedding_proj(self.embedding.weight)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(quant_codebook**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(quant_codebook, "n d -> d n"))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            commit_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b h c -> b c h").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3]
            )

        return (z_q, torch.tensor(0.0), min_encoding_indices), LossBreakdown(
            torch.tensor(0.0), torch.tensor(0.0), commit_loss, torch.tensor(0.0)
        )


# distributed helpers


@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t


# helper functions


def exists(v):
    return v is not None


def identity(t):
    return t


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def l2norm(t):
    return F.normalize(t, dim=-1)


# entropy


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


# cosine sim linear


class CosineSimLinear(Module):
    def __init__(self, dim_in, dim_out, scale=1.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=0)
        return (x @ w) * self.scale


# borrowed from https://github.com/lucidrains/vector-quantize-pytorch/blob/8c27a0e42c983ad4b7e54e5df35f0b9b8f332db9/vector_quantize_pytorch/lookup_free_quantization.py#L96
class LFQ(Module):
    def __init__(
        self,
        *,
        dim=None,
        codebook_size=None,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.0,
        diversity_gamma=1.0,
        straight_through_activation=nn.Identity(),
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        codebook_scale=1.0,  # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy=1.0,  # make less than 1. to only use a random fraction of the probs for per sample entropy
        has_projections=None,
        projection_has_bias=True,
        soft_clamp_input_value=None,
        cosine_sim_project_in=False,
        cosine_sim_project_in_scale=None,
        channel_first=None,
        experimental_softplus_entropy_loss=False,
        entropy_loss_offset=5.0,  # how much to shift the loss before softplus
        spherical=False,  # from https://arxiv.org/abs/2406.07548
        force_quantization_f32=True,  # will force the quantization step to be full precision
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(
            codebook_size
        ), "either dim or codebook_size must be specified for LFQ"
        assert (
            not exists(codebook_size) or log2(codebook_size).is_integer()
        ), f"your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})"

        codebook_size = default(codebook_size, lambda: 2**dim)
        self.codebook_size = self.n_e = codebook_size

        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale=cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias=projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = (
            nn.Linear(codebook_dims, dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first

        self.channel_first = channel_first

        # straight through activation

        self.activation = straight_through_activation

        # whether to use BSQ (binary spherical quantization)

        self.spherical = spherical
        self.maybe_l2norm = (lambda t: l2norm(t) * self.codebook_scale) if spherical else identity

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.0
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # whether to make the entropy loss positive through a softplus (experimental, please report if this worked or not in discussions)

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference

        self.register_buffer("mask", 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # whether to force quantization step to be f32

        self.force_quantization_f32 = force_quantization_f32

        # codes

        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer("codebook", codebook.float(), persistent=False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(self, indices, project_out=True):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... -> ... 1")

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = self.maybe_l2norm(codes)

        codes = rearrange(codes, "... c d -> ... (c d)")

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if should_transpose:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(
        self,
        x,
        inv_temperature=100.0,
        return_loss_breakdown=False,
        mask=None,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)

        if should_transpose:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack_one(x, "b * d")

        assert (
            x.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but received {x.shape[-1]}"

        x = self.project_in(x)

        # maybe soft clamp

        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        # split out number of codebooks

        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)

        # maybe l2norm

        x = self.maybe_l2norm(x)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        quantization_context = (
            partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext
        )

        with quantization_context():
            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            # quantize by eq 3.

            original_input = x

            codebook_value = torch.ones_like(x) * self.codebook_scale
            quantized = torch.where(x > 0, codebook_value, -codebook_value)

            # calculate indices

            indices = reduce((quantized > 0).int() * self.mask.int(), "b n c d -> b n c", "sum")

            # maybe l2norm

            quantized = self.maybe_l2norm(quantized)

            # use straight-through gradients (optionally with custom activation fn) if training

            if self.training:
                x = self.activation(x)
                x = x + (quantized - x).detach()
            else:
                x = quantized

            # entropy aux loss

            if self.training:
                if force_f32:
                    codebook = self.codebook.float()

                codebook = self.maybe_l2norm(codebook)

                # whether to only use a fraction of probs, for reducing memory

                if self.frac_per_sample_entropy < 1.0:
                    # account for mask
                    if exists(mask):
                        original_input = original_input[mask]
                    original_input = rearrange(original_input, "b n ... -> (b n) ...")

                    num_tokens = original_input.size(0)
                    num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
                    rand_mask = torch.randn(num_tokens).argsort(dim=-1) < num_sampled_tokens

                    sampled_input = original_input[rand_mask]

                    sampled_distance = -2 * einsum(
                        "... i d, j d -> ... i j", sampled_input, codebook
                    )

                    sampled_prob = (-sampled_distance * inv_temperature).softmax(dim=-1)

                    per_sample_probs = sampled_prob
                else:
                    if exists(mask):
                        original_input = original_input[mask]
                    original_input = rearrange(original_input, "b n ... -> (b n) ...")
                    # the same as euclidean distance up to a constant
                    distance = -2 * einsum("... i d, j d -> ... i j", original_input, codebook)

                    prob = (-distance * inv_temperature).softmax(dim=-1)

                    per_sample_probs = prob

                # calculate per sample entropy

                per_sample_entropy = entropy(per_sample_probs).mean()

                # distribution over all available tokens in the batch

                avg_prob = reduce(per_sample_probs, "... c d -> c d", "mean")

                avg_prob = maybe_distributed_mean(avg_prob)

                codebook_entropy = entropy(avg_prob).mean()

                # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
                # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

                entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
            else:
                # if not training, just return dummy 0
                entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

            # whether to make the entropy loss positive or not through a (shifted) softplus

            if self.training and self.experimental_softplus_entropy_loss:
                entropy_aux_loss = F.softplus(entropy_aux_loss + self.entropy_loss_offset)

            # commit loss

            if self.training and self.commitment_loss_weight > 0.0:
                commit_loss = F.mse_loss(original_input, quantized.detach(), reduction="none")

                if exists(mask):
                    commit_loss = commit_loss[mask]

                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            # input back to original dtype if needed

            if force_f32:
                x = x.type(orig_dtype)

        # merge back codebook dim

        x = rearrange(x, "b n c d -> b n (c d)")

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if should_transpose:
            x = unpack_one(x, ps, "b * d")
            x = rearrange(x, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        # complete aux loss

        aux_loss = (
            entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight
        )

        # returns

        if not return_loss_breakdown:
            return (x, aux_loss, indices)

        return (x, aux_loss, indices), LossBreakdown(
            per_sample_entropy, codebook_entropy, commit_loss, torch.tensor(0.0)
        )


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks=1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices=True,
        force_quantization_f32=True,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            self.n_e = self.codebook_size
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings"""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = (
            partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext
        )

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, "b n c d -> b n (c d)")

            codes = codes.type(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")

            indices = maybe(unpack_one)(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, "... 1 -> ...")

        # return quantized output and indices

        aux_loss = torch.tensor(0.0)
        return (out, aux_loss, indices), LossBreakdown(
            torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        )
