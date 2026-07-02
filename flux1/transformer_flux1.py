# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ---------------------------------------------------------------------------
# SEGA — Spectral-Energy Guided Attention (reverse / sharpness-biased)
#
# Compact implementation: hardcodes direction=-1 (content-complementing),
# score_source="spectral", and supports only power_res / log_res base formulas.
#
#   z_k = standardise( log E[bin(k)] )
#   s_k = tanh( beta * z_k )
#   m_k = base_mscale * ( 1 - alpha * spread * tanh(beta * z_k) )
#
# High spectral-energy dims get LOWER m_k (sharpness-biased / reverse).
# ---------------------------------------------------------------------------

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle


logger = logging.get_logger(__name__)

@torch.no_grad()
def _resample_1d(y: torch.Tensor, n: int) -> torch.Tensor:
    """Resample a 1D tensor to length n using linear interpolation."""
    if y.ndim != 1:
        y = y.flatten()
    if int(y.numel()) == int(n):
        return y
    y2 = y[None, None, :].float()
    out = F.interpolate(y2, size=int(n), mode="linear", align_corners=True)
    return out[0, 0]


@torch.no_grad()
def compute_axis_spectral_profiles_continuous(
    hidden_states: torch.Tensor, *, height: int, width: int, n_freq: int = 128
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Continuous normalized frequency profiles (0..1) for each axis, matching flux-ntk/flux-yarn.
    """
    B, S, C = hidden_states.shape
    n_spatial = min(int(S), int(height) * int(width))
    spatial = hidden_states[:, :n_spatial].reshape(B, height, width, C)
    sm = spatial.float().mean(dim=(0, -1))
    sm = sm - sm.mean()

    def _axis_profile(sm_: torch.Tensor, *, axis: int) -> torch.Tensor:
        fft = torch.fft.rfft(sm_, dim=axis)
        power = fft.abs().pow(2)
        power = power.mean(dim=1 - axis)
        return power.float()

    e_h = _axis_profile(sm, axis=0)
    e_w = _axis_profile(sm, axis=1)

    e_h = _resample_1d(e_h, int(n_freq))
    e_w = _resample_1d(e_w, int(n_freq))
    freq = torch.linspace(0.0, 1.0, int(n_freq), device=e_h.device, dtype=torch.float32)
    return {"height": {"freq": freq, "energy": e_h}, "width": {"freq": freq, "energy": e_w}}


@torch.no_grad()
def compute_isotropic_spectral_profile_continuous(
    hidden_states: torch.Tensor, *, height: int, width: int, n_freq: int = 128
) -> Dict[str, torch.Tensor]:
    """
    Continuous normalized radial frequency profile (0..1), matching flux-ntk/flux-yarn.
    """
    B, S, C = hidden_states.shape
    n_spatial = min(int(S), int(height) * int(width))
    spatial = hidden_states[:, :n_spatial].reshape(B, height, width, C)
    sm = spatial.float().mean(dim=(0, -1))
    sm = sm - sm.mean()

    power = torch.fft.fftshift(torch.fft.fft2(sm)).abs().pow(2).float()

    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    y = torch.arange(height, device=power.device, dtype=torch.float32) - cy
    x = torch.arange(width, device=power.device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(yy**2 + xx**2)
    r = r / (r.max().clamp(min=1e-8))

    n_bins = max(int(min(height, width) // 2), 8)
    bin_idx = (r * n_bins).long().clamp(0, n_bins - 1).flatten()
    flat_pw = power.flatten()

    energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_sum.scatter_add_(0, bin_idx, flat_pw)
    energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(flat_pw))
    e = energy_sum / (energy_cnt + 1e-8)

    e = _resample_1d(e, int(n_freq))
    freq = torch.linspace(0.0, 1.0, int(n_freq), device=e.device, dtype=torch.float32)
    return {"freq": freq, "energy": e}


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> torch.Tensor:
    if use_real:
        cos, sin = freqs_cis
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")
        cos, sin = cos.to(x.device), sin.to(x.device)
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


# ---------------------------------------------------------------------------
# base_mscale — only power_res and log_res
# ---------------------------------------------------------------------------

def compute_base_mscale(
    target_res: int,
    training_res: int,
    formula: str = "power_res",
    coefficient: Optional[float] = None,
    resolution_ratio_override: Optional[float] = None,
) -> float:
    if resolution_ratio_override is not None:
        s = max(float(resolution_ratio_override), 1.0)
    else:
        s = max(float(target_res) / float(training_res), 1.0)
    if formula == "power_res":
        c = 0.1 if coefficient is None else float(coefficient)
        return s ** c
    if formula == "log_res":
        c = 0.1 if coefficient is None else float(coefficient)
        return 1.0 + c * math.log(s)
    raise ValueError(f"Unknown base_mscale formula: {formula!r}. Use 'power_res' or 'log_res'.")


# ---------------------------------------------------------------------------
# YaRN helpers — cloned from base/flux-yarn
# ---------------------------------------------------------------------------

def find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    """Inverse dim formula to find number of rotations for YaRN."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    """Find the correction range for YaRN interpolation."""
    low = np.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = np.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_mask(min_val, max_val, dim):
    """Create linear ramp mask for frequency blending."""
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def find_newbase_ntk(dim, base, scale):
    """Calculate the new base for NTK-aware scaling."""
    return base * (scale ** (dim / (dim - 2)))


# ---------------------------------------------------------------------------
# SEGA mscale — reverse spectral, symmetric multiplicative
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sega_allocation(
    energy_profile: torch.Tensor,
    freqs: torch.Tensor,
    base_mscale: float,
    spread: float,
    alpha: float = 0.15,
    beta: float = 1.5,
    min_mscale: float = 1.0,
    max_mscale: Optional[float] = None,
) -> torch.Tensor:
    """
    Spectral-Energy Guided Attention (SEGA): per-dim RoPE mscale (asymmetric band).

        z_k = standardise( log E[bin(k)] )
        s_k = tanh( beta * z_k ),  re-centred
        m_k = base_mscale * ( 1 - alpha * spread * s_k )

    High spectral-energy dims get lower m_k (sharpness-biased).
    """
    D_half = freqs.shape[0]
    eps = 1e-8

    if spread <= 0.0 or alpha <= 0.0:
        return torch.full((D_half,), float(base_mscale), device=freqs.device, dtype=torch.float32)

    # Map each RoPE dim to its FFT bin via log-period
    periods = 2.0 * math.pi / freqs.clamp(min=eps)
    log_periods = torch.log(periods)
    min_lp, max_lp = log_periods.min(), log_periods.max()
    if (max_lp - min_lp).item() > 1e-6:
        lp_norm = (log_periods - min_lp) / (max_lp - min_lp)  # 0=high-freq, 1=low-freq
    else:
        lp_norm = torch.zeros_like(log_periods)

    n_bins = energy_profile.shape[0]
    bin_pos = (1.0 - lp_norm) * (n_bins - 1)
    j_low = bin_pos.floor().long().clamp(0, n_bins - 1)
    j_high = (j_low + 1).clamp(0, n_bins - 1)
    frac = (bin_pos - j_low.to(bin_pos.dtype)).clamp(0.0, 1.0)

    E = energy_profile.to(freqs.device).clamp(min=eps)
    log_E = torch.log(E)
    raw = log_E[j_low] * (1.0 - frac) + log_E[j_high] * frac

    # Standardise + tanh + re-centre
    z = raw - raw.mean()
    z = z / z.std().clamp(min=eps)
    s = torch.tanh(float(beta) * z)
    s = s - s.mean()

    # direction = -1  =>  subtract
    m = float(base_mscale) * (1.0 - float(alpha) * float(spread) * s)
    m = m.clamp(min=float(min_mscale))
    if max_mscale is not None:
        m = m.clamp(max=float(max_mscale))
    return m.to(torch.float32)


def compute_sega_excess_blend(
    window_size: Optional[int],
    window_floor: int,
    max_patches: int,
    rope_method: str = "sega",
) -> float:
    """HSWA locality factor for SEGA mscale excess (Flux RoPE).

    Per-frequency effective mscale uses ``m_eff = 1 + λ * (m - 1)`` so the
    floor at 1 is preserved; only the deviation above 1 is scaled.

    Returns λ in ``[0, 1]``:

    - Global attention (``window_size is None``): λ = 1 (full SEGA excess).
    - ``rope_method != "sega"`` or missing HSWA metadata: λ = 1 (unchanged).
    - Local attention at ``window_size == window_floor``: λ = 0 (no excess).
    - Linear between ``window_floor`` and ``max_patches`` in patch units.
    """
    if rope_method != "sega":
        return 1.0
    if window_size is None:
        return 1.0
    wf = int(window_floor)
    mp = int(max_patches)
    if wf <= 0 or mp <= 0:
        return 1.0
    denom = float(mp - wf)
    if denom <= 0.0:
        return 0.0
    lam = (float(window_size) - float(wf)) / denom
    return max(0.0, min(1.0, lam))


# ---------------------------------------------------------------------------
# 1-D RoPE embedding with SEGA per-dim mscale
# ---------------------------------------------------------------------------

def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real: bool = True,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    freqs_dtype=torch.float32,
    # SEGA knobs
    energy_profile: Optional[torch.Tensor] = None,
    mscale_spread: Optional[float] = None,
    mscale_alpha: float = 0.15,
    mscale_beta: float = 1.5,
    mscale_min: float = 1.0,
    # base_mscale
    base_mscale_formula: str = "power_res",
    base_mscale_coefficient: Optional[float] = None,
    target_res: Optional[int] = None,
    training_res: Optional[int] = None,
    # visualization
    viz_collector=None,
    viz_step: int = -1,
    viz_axis: Optional[str] = None,
    disable_mscale: bool = False,
    sega_base_mscale_res_ratio: Optional[float] = None,
    # HSWA: blend only the excess over 1 for per-freq SEGA mscale (see compute_sega_excess_blend).
    sega_excess_blend: float = 1.0,
    mscale_ceiling: Optional[float] = None,
    # YaRN
    yarn: bool = False,
    max_pe_len: Optional[Union[int, torch.Tensor]] = None,
    ori_max_pe_len: int = 64,
):
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    dim_indices = torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)
    inv_freq = 1.0 / (theta ** (dim_indices / dim))

    effective_mscale: Union[float, torch.Tensor] = 1.0

    max_pe_len_value = (
        float(max_pe_len.item()) if isinstance(max_pe_len, torch.Tensor) else (float(max_pe_len) if max_pe_len is not None else None)
    )

    if yarn and max_pe_len_value is not None and max_pe_len_value > float(ori_max_pe_len):
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len, dtype=freqs_dtype, device=pos.device)
        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

        # YaRN parameters from base/flux-yarn (fixed, no DyPE).
        beta_0 = 1.25
        beta_1 = 0.75
        gamma_0 = 16
        gamma_1 = 2

        freqs_base = inv_freq
        freqs_linear = 1.0 / torch.einsum(
            "..., f -> ... f",
            scale,
            (theta ** (dim_indices / dim)),
        )

        new_base = find_newbase_ntk(dim, theta, scale)
        if new_base.dim() > 0:
            new_base = new_base.view(-1, 1)
        freqs_ntk = 1.0 / torch.pow(new_base, (dim_indices / dim))
        if freqs_ntk.dim() > 1:
            freqs_ntk = freqs_ntk.squeeze()

        low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)
        freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(pos.device).to(freqs_dtype))
        freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask

        low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)
        freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(pos.device).to(freqs_dtype))
        freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask

        effective_mscale = float(torch.where(
            scale <= 1.0,
            torch.tensor(1.0, dtype=freqs_dtype, device=pos.device),
            0.1 * torch.log(scale) + 1.0,
        ).item())
    elif ntk_factor > 1.0:
        scaled_theta = theta * ntk_factor
        freqs = 1.0 / (scaled_theta ** (dim_indices / dim))

        if not disable_mscale:
            base_ms = compute_base_mscale(
                target_res=target_res,
                training_res=training_res,
                formula=base_mscale_formula,
                coefficient=base_mscale_coefficient,
                resolution_ratio_override=sega_base_mscale_res_ratio,
            )
        else:
            base_ms = 1.0

        use_sega = (
            energy_profile is not None
            and mscale_spread is not None
            and float(mscale_spread) > 0.0
            and base_ms > 1.0 + 1e-8
        )

        if use_sega:
            effective_mscale = compute_sega_allocation(
                energy_profile=energy_profile,
                freqs=freqs,
                base_mscale=base_ms,
                spread=float(mscale_spread),
                alpha=float(mscale_alpha),
                beta=float(mscale_beta),
                min_mscale=float(mscale_min),
                max_mscale=mscale_ceiling,
            )
        else:
            effective_mscale = base_ms
    elif ntk_factor <= 1.0 and sega_base_mscale_res_ratio is not None:
        scaled_theta = theta * max(float(ntk_factor), 1.0)
        freqs = 1.0 / (scaled_theta ** (dim_indices / dim))
        if not disable_mscale:
            base_ms = compute_base_mscale(
                target_res=target_res,
                training_res=training_res,
                formula=base_mscale_formula,
                coefficient=base_mscale_coefficient,
                resolution_ratio_override=sega_base_mscale_res_ratio,
            )
            effective_mscale = base_ms
        else:
            freqs = inv_freq
    else:
        freqs = inv_freq

    if isinstance(effective_mscale, torch.Tensor):
        lam = float(sega_excess_blend)
        if lam != 1.0:
            effective_mscale = 1.0 + lam * (effective_mscale - 1.0)

    # Visualization: record per-frequency mscale vs log(period) (downsampled) when SEGA is active.
    if viz_collector is not None and viz_step >= 0 and viz_axis is not None:
        try:
            eps = 1e-8
            periods = (2.0 * math.pi) / freqs.clamp(min=eps)
            log_periods = torch.log(periods)
            n_pts = min(64, int(log_periods.shape[0]))
            idx = torch.linspace(0, max(log_periods.shape[0] - 1, 0), steps=n_pts, device=log_periods.device).long()
            if isinstance(effective_mscale, torch.Tensor):
                viz_collector.record_mscale(
                    step=int(viz_step),
                    axis=str(viz_axis),
                    log_period=log_periods[idx].detach().cpu().tolist(),
                    mscale=effective_mscale[idx].detach().cpu().float().tolist(),
                    min=float(effective_mscale.min().item()),
                    mean=float(effective_mscale.mean().item()),
                    max=float(effective_mscale.max().item()),
                )

            # FFT-energy: map axis spectral energy onto the RoPE log-period grid.
            if energy_profile is not None and hasattr(viz_collector, "record_fft_heatmap"):
                E = energy_profile.to(freqs.device).float().clamp(min=eps)
                min_lp, max_lp = log_periods.min(), log_periods.max()
                if (max_lp - min_lp).item() > 1e-6:
                    lp_norm = (log_periods - min_lp) / (max_lp - min_lp)
                else:
                    lp_norm = torch.zeros_like(log_periods)
                n_bins = int(E.shape[0])
                bin_pos = (1.0 - lp_norm) * (n_bins - 1)
                j_low = bin_pos.floor().long().clamp(0, n_bins - 1)
                j_high = (j_low + 1).clamp(0, n_bins - 1)
                frac = (bin_pos - j_low.to(bin_pos.dtype)).clamp(0.0, 1.0)
                energy_vals = E[j_low] * (1.0 - frac) + E[j_high] * frac
                viz_collector.record_fft_heatmap(
                    step=int(viz_step),
                    axis=str(viz_axis),
                    log_period=log_periods[idx].detach().cpu().float().tolist(),
                    energy=energy_vals[idx].detach().cpu().float().tolist(),
                    min=float(energy_vals.min().item()),
                    mean=float(energy_vals.mean().item()),
                    max=float(energy_vals.max().item()),
                )
        except Exception:
            pass

    freqs = torch.outer(pos.float(), freqs)

    if freqs.device.type == "npu":
        freqs = freqs.float()

    if isinstance(effective_mscale, torch.Tensor):
        if use_real and repeat_interleave_real:
            ms = effective_mscale.repeat_interleave(2)
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() * ms
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() * ms
        elif use_real:
            ms = torch.cat([effective_mscale, effective_mscale], dim=-1)
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float() * ms
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float() * ms
        else:
            freqs_cis = torch.polar(effective_mscale.unsqueeze(0).expand_as(freqs), freqs)
            return freqs_cis
    else:
        if use_real and repeat_interleave_real:
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() * effective_mscale
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() * effective_mscale
        elif use_real:
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float() * effective_mscale
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float() * effective_mscale
        else:
            freqs_cis = torch.polar(torch.ones_like(freqs) * effective_mscale, freqs)
            return freqs_cis

    return freqs_cos, freqs_sin


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_spectral_energy_profile(hidden_states, height, width, n_bins):
    B, S, C = hidden_states.shape
    n_spatial = min(S, height * width)
    spatial = hidden_states[:, :n_spatial].reshape(B, height, width, C)
    spatial_map = spatial.float().mean(dim=(0, -1))
    spatial_map = spatial_map - spatial_map.mean()

    power = torch.fft.fftshift(torch.fft.fft2(spatial_map)).abs().pow(2)

    cy, cx = height / 2.0, width / 2.0
    y = torch.arange(height, device=power.device, dtype=torch.float32) - cy
    x = torch.arange(width, device=power.device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius_norm = torch.sqrt(yy ** 2 + xx ** 2)
    radius_norm = radius_norm / (radius_norm.max() + 1e-8)

    bin_idx = (radius_norm * n_bins).long().clamp(0, n_bins - 1).flatten()
    flat_pw = power.flatten()

    energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_sum.scatter_add_(0, bin_idx, flat_pw)
    energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(flat_pw))
    return energy_sum / (energy_cnt + 1e-8)


@torch.no_grad()
def compute_axis_spectral_profiles(hidden_states, height, width, n_bins_h, n_bins_w):
    B, S, C = hidden_states.shape
    n_spatial = min(S, height * width)
    spatial = hidden_states[:, :n_spatial].reshape(B, height, width, C)
    sm = spatial.float().mean(dim=(0, -1))
    sm = sm - sm.mean()

    def _axis_profile(sm, axis, n_bins, length):
        fft = torch.fft.fft(sm, dim=axis)
        power = fft.abs().pow(2).mean(dim=1 - axis)
        half = length // 2 + 1
        power = power[:half]
        freq_norm = torch.linspace(0.0, 1.0, half, device=power.device)
        bin_idx = (freq_norm * n_bins).long().clamp(0, n_bins - 1)
        energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_sum.scatter_add_(0, bin_idx, power.float())
        energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(power, dtype=torch.float32))
        return energy_sum / (energy_cnt + 1e-8)

    return (
        _axis_profile(sm, axis=0, n_bins=n_bins_h, length=height),
        _axis_profile(sm, axis=1, n_bins=n_bins_w, length=width),
    )


@torch.no_grad()
def compute_dynamic_spread(energy_profile, spread_min=0.0, spread_max=1.0, alpha=1.5):
    eps = 1e-8
    energy = energy_profile.clamp(min=eps)
    geo_mean = torch.exp(torch.log(energy).mean())
    arith_mean = energy.mean()
    flatness = (geo_mean / (arith_mean + eps)).clamp(0.0, 1.0)
    concentration = 1.0 - flatness.item()
    return spread_min + (spread_max - spread_min) * (1.0 - (1.0 - concentration) ** alpha)


@torch.no_grad()
def _spatial_grid_from_img_hidden(
    hidden_img: torch.Tensor,
    img_ids: torch.Tensor,
) -> Tuple[torch.Tensor, int, int]:
    """Scatter image tokens [B, N_img, C] into [B, img_h, img_w, C] using id coordinates."""
    row_min = img_ids[:, 1].min()
    col_min = img_ids[:, 2].min()
    rows = (img_ids[:, 1] - row_min).long()
    cols = (img_ids[:, 2] - col_min).long()
    img_h = int(rows.max().item()) + 1
    img_w = int(cols.max().item()) + 1
    B, N, C = hidden_img.shape
    spatial = hidden_img.new_zeros(B, img_h, img_w, C)
    spatial[:, rows, cols, :] = hidden_img
    return spatial, img_h, img_w


@torch.no_grad()
def _crop_window_spatial_hidden(
    hidden_img: torch.Tensor,
    img_ids: torch.Tensor,
    win_idx: torch.Tensor,
) -> Tuple[torch.Tensor, int, int]:
    """Minimal axis-aligned crop of the latent patch covered by ``win_idx``."""
    dev = hidden_img.device
    img_ids = img_ids.to(dev)
    win_idx = win_idx.to(dev)
    spatial, _, _ = _spatial_grid_from_img_hidden(hidden_img, img_ids)
    rows = img_ids[win_idx, 1]
    cols = img_ids[win_idx, 2]
    row_min = img_ids[:, 1].min()
    col_min = img_ids[:, 2].min()
    r0 = int((rows.min() - row_min).item())
    r1 = int((rows.max() - row_min).item())
    c0 = int((cols.min() - col_min).item())
    c1 = int((cols.max() - col_min).item())
    crop = spatial[:, r0 : r1 + 1, c0 : c1 + 1, :].contiguous()
    return crop, crop.shape[1], crop.shape[2]


@torch.no_grad()
def _record_isotropic_fft_heatmap(
    *,
    viz_collector,
    viz_step: int,
    freqs: torch.Tensor,
    iso_profile: torch.Tensor,
):
    """Record isotropic FFT energy mapped to the RoPE log-period grid."""
    if viz_collector is None or viz_step < 0:
        return
    if not hasattr(viz_collector, "record_fft_iso_heatmap"):
        return
    try:
        eps = 1e-8
        periods = (2.0 * math.pi) / freqs.clamp(min=eps)
        log_periods = torch.log(periods)
        n_pts = min(64, int(log_periods.shape[0]))
        idx = torch.linspace(0, max(log_periods.shape[0] - 1, 0), steps=n_pts, device=log_periods.device).long()

        E = iso_profile.to(freqs.device).float().clamp(min=eps)
        min_lp, max_lp = log_periods.min(), log_periods.max()
        if (max_lp - min_lp).item() > 1e-6:
            lp_norm = (log_periods - min_lp) / (max_lp - min_lp)
        else:
            lp_norm = torch.zeros_like(log_periods)
        n_bins = int(E.shape[0])
        bin_pos = (1.0 - lp_norm) * (n_bins - 1)
        j_low = bin_pos.floor().long().clamp(0, n_bins - 1)
        j_high = (j_low + 1).clamp(0, n_bins - 1)
        frac = (bin_pos - j_low.to(bin_pos.dtype)).clamp(0.0, 1.0)
        energy_vals = E[j_low] * (1.0 - frac) + E[j_high] * frac

        viz_collector.record_fft_iso_heatmap(
            step=int(viz_step),
            axis="isotropic",
            log_period=log_periods[idx].detach().cpu().float().tolist(),
            energy=energy_vals[idx].detach().cpu().float().tolist(),
            min=float(energy_vals.min().item()),
            mean=float(energy_vals.mean().item()),
            max=float(energy_vals.max().item()),
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HSWA — sliding-window geometry helpers
#
# Re-added on top of sega/flux_new (verbatim from base/flux-hswa).  These are
# pure CPU-side window-partition / mask helpers that are independent of any
# sega state, so they slot in without changing the sega code paths.
# ---------------------------------------------------------------------------

ROPE_METHODS = ("ntk", "sega", "none", "yarn", "dype")


def build_sliding_window_mask(
    img_ids: torch.Tensor,
    txt_len: int,
    window_size: int,
    overlap_padding: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """Build an additive attention mask enforcing a 2-D sliding window.

    Sequence layout: [txt_tokens | img_tokens]

    Text tokens attend to everything.  Image tokens attend only within their
    sliding window plus overlap_padding, and (by default) to all text tokens.
    Per-window text caps are applied only in the chunked (``--optimized``) path.

    Built entirely on CPU with integer arithmetic.  Stored as bfloat16 to
    halve GPU memory footprint vs float32.  Only the final tensor is moved
    to device.
    """
    N_img = img_ids.shape[0]
    total_len = txt_len + N_img
    # window_size is the maximum relative distance (not a diameter to halve).
    effective_half = window_size + overlap_padding

    rows = img_ids[:, 1].short().cpu()
    cols = img_ids[:, 2].short().cpu()

    row_dist = (rows.unsqueeze(0) - rows.unsqueeze(1)).abs()
    col_dist = (cols.unsqueeze(0) - cols.unsqueeze(1)).abs()
    blocked = (row_dist > effective_half) | (col_dist > effective_half)

    mask = torch.zeros(total_len, total_len, dtype=torch.bfloat16)
    mask[txt_len:, txt_len:].masked_fill_(blocked, float("-inf"))

    return mask.unsqueeze(0).unsqueeze(0).to(device=device)


# Native training extent in img_ids patch units (matches ``run_flux.BASE_PATCHES``).
_LOCAL_ORIGIN_BASE_PATCHES = 64
_LOCAL_ORIGIN_ROPE_DIM = 56


def _ntk_factor_for_window_patches(window_patches: int) -> float:
    """NTK factor for a canvas of ``window_patches`` per side — not full 4K schedule."""
    wp = int(window_patches)
    if wp <= _LOCAL_ORIGIN_BASE_PATCHES:
        return 1.0
    exponent = 2 * _LOCAL_ORIGIN_ROPE_DIM / (_LOCAL_ORIGIN_ROPE_DIM - 2)
    return (wp / float(_LOCAL_ORIGIN_BASE_PATCHES)) ** exponent


def _uniform_window_tiles(
    img_ids: torch.Tensor,
    window_size: int,
    overlap_padding: int = 0,
) -> bool:
    """True when every window is a full ``window_size`` square (no overlap expansion)."""
    if int(overlap_padding) != 0:
        return False
    win_indices = _build_window_index(img_ids, window_size, overlap_padding=0)
    ws2 = int(window_size) * int(window_size)
    return bool(win_indices) and all(idx.shape[0] == ws2 for idx in win_indices)


def _build_img_local_rope_index(img_ids: torch.Tensor, window_size: int) -> torch.Tensor:
    """Map each image token to its slot in a ``window_size``×``window_size`` local RoPE table."""
    ws = int(window_size)
    # Integer arithmetic: low-precision id dtypes (bf16) round positions above 256.
    rows = img_ids[:, 1].float().round().long()
    cols = img_ids[:, 2].float().round().long()
    rows = rows - rows.min()
    cols = cols - cols.min()
    return (rows % ws) * ws + (cols % ws)


def compute_window_local_rope_emb(
    *,
    pos_embed,
    img_ids: torch.Tensor,
    win_idx: torch.Tensor,
    n_text: int,
    window_size: int,
    rope_method: str = "none",
    spread_min: float = 0.0,
    spread_max: float = 1.0,
    spread_alpha: float = 1.5,
    hidden_states: Optional[torch.Tensor] = None,
    txt_len: int = 0,
    is_dual_stream: bool = False,
    leaf_i: int = 0,
    dwt_leaf_sizes: Optional[List[int]] = None,
    leaf_ntk_factors: Optional[List[float]] = None,
    leaf_sega_blends: Optional[List[float]] = None,
    window_floor: int = 0,
    max_patches: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin for window-local RoPE: text at (0,0), image patches from window origin."""
    dev = win_idx.device
    txt_local, img_local = _build_window_local_ids(
        img_ids, win_idx, n_text, dev, img_ids.dtype,
    )
    ids_w = torch.cat([txt_local, img_local], dim=0)

    spread = None
    eh = ew = None
    if rope_method == "sega":
        if hidden_states is None:
            raise ValueError("window-local SEGA RoPE requires hidden_states")
        if is_dual_stream:
            img_hidden = hidden_states
        else:
            img_hidden = hidden_states[:, txt_len:, :]
        crop, h, w = _crop_window_spatial_hidden(img_hidden, img_ids, win_idx)
        crop_seq = crop.reshape(crop.shape[0], h * w, -1)
        n_bins_h = max(h // 2, 1)
        n_bins_w = max(w // 2, 1)
        n_bins_iso = max(max(h, w) // 2, 1)
        eh, ew = compute_axis_spectral_profiles(crop_seq, h, w, n_bins_h, n_bins_w)
        iso = compute_spectral_energy_profile(crop_seq, h, w, n_bins_iso)
        spread = compute_dynamic_spread(
            iso, spread_min=spread_min, spread_max=spread_max, alpha=spread_alpha,
        )

    dwt_leaf_sizes = dwt_leaf_sizes or []
    leaf_ntk_factors = leaf_ntk_factors or []
    leaf_sega_blends = leaf_sega_blends or []
    ws_eff = (
        int(dwt_leaf_sizes[leaf_i])
        if dwt_leaf_sizes and leaf_i < len(dwt_leaf_sizes)
        else int(window_size)
    )
    if dwt_leaf_sizes and leaf_i < len(leaf_ntk_factors):
        local_ntk = float(leaf_ntk_factors[leaf_i])
    else:
        local_ntk = _ntk_factor_for_window_patches(ws_eff)
    if rope_method == "sega":
        if dwt_leaf_sizes and leaf_i < len(leaf_sega_blends):
            lam = float(leaf_sega_blends[leaf_i])
        else:
            lam = float(compute_sega_excess_blend(
                ws_eff, window_floor, max_patches, rope_method="sega",
            ))
    else:
        lam = 1.0

    local_h = int(img_local[:, 1].max().item()) + 1
    local_w = int(img_local[:, 2].max().item()) + 1
    local_target_res = max(local_h, local_w) * 16
    pe = pos_embed
    old_viz = getattr(pe, "_viz_collector", None)
    pe._viz_collector = None
    old_disable = getattr(pe, "_disable_mscale", False)
    if rope_method != "sega":
        pe._disable_mscale = True
    try:
        return pe(
            ids_w,
            ntk_factor=float(local_ntk),
            mscale_spread=spread,
            energy_profile_h=eh,
            energy_profile_w=ew,
            target_res=local_target_res,
            sega_excess_blend=float(lam),
        )
    finally:
        pe._viz_collector = old_viz
        pe._disable_mscale = old_disable


def compute_window_text_cap(
    window_size: int,
    window_floor: int,
    max_patches: int,
    txt_len: int,
    min_fraction: Optional[float],
) -> Optional[int]:
    """Text-token cap from window size (linear in patch units).

  At ``window_size >= max_patches`` (largest HSWA windows): all ``txt_len`` tokens.
  At ``window_size == window_floor`` (smallest): ``round(txt_len * min_fraction)``.
  Between: linear blend of kept fraction from 1.0 down to ``min_fraction``.
    """
    if min_fraction is None:
        return None
    min_fraction = float(min_fraction)
    if min_fraction >= 1.0:
        return None
    if min_fraction <= 0.0:
        raise ValueError("min_fraction must be in (0, 1)")
    ws = int(window_size)
    wf = int(window_floor)
    mp = int(max_patches) if int(max_patches) > 0 else ws
    t_len = int(txt_len)
    if ws >= mp:
        return None
    t = (mp - ws) / max(1, mp - wf)
    t = max(0.0, min(1.0, t))
    frac = 1.0 - t * (1.0 - min_fraction)
    cap = max(1, int(round(t_len * frac)))
    if cap >= t_len:
        return None
    return cap


def _select_window_text_indices(
    q_win: torch.Tensor,
    k_txt: torch.Tensor,
    cap: int,
) -> Optional[torch.Tensor]:
    """Pick text token indices with highest Q·K affinity to this spatial window.

    Returns ``None`` when all text tokens are kept. Indices are sorted ascending
    so RoPE / sequence order stay stable.
    """
    cap = int(cap)
    if cap <= 0:
        return None
    t_len = int(k_txt.shape[1])
    k = min(cap, t_len)
    if k >= t_len:
        return None
    # [B, W, H, D] × [B, T, H, D] → max over window positions, mean over heads
    scores = torch.einsum("bwhd,bthd->bwth", q_win, k_txt).amax(dim=1).mean(dim=-1)
    idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices
    return torch.sort(idx, dim=1).values


def _gather_seq_dim1(
    tensor: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather ``tensor`` [B, S, …] along dim 1 with per-batch ``indices`` [B, K]."""
    b, k = indices.shape
    batch = torch.arange(b, device=tensor.device)[:, None].expand(b, k)
    return tensor[batch, indices]


def _build_window_local_ids(
    img_ids: torch.Tensor,
    win_idx: torch.Tensor,
    n_text: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Window-local ``txt_ids`` at origin and image ids relative to window top-left."""
    win = img_ids.to(device=device)[win_idx]
    r0 = win[:, 1].min()
    c0 = win[:, 2].min()
    img_local = win.clone()
    img_local[:, 1] = win[:, 1] - r0
    img_local[:, 2] = win[:, 2] - c0
    txt_local = torch.zeros(n_text, 3, device=device, dtype=dtype)
    if n_text > 0:
        txt_local[:, 0] = win[0, 0]
    return txt_local, img_local


def _unrotate_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Undo RoPE (conjugate rotation) on Q/K along sequence dim 1."""
    return (
        apply_rotary_emb(q, (cos, -sin), sequence_dim=1),
        apply_rotary_emb(k, (cos, -sin), sequence_dim=1),
    )


def _starts_axis(total: int, w: int) -> list:
    """Edge-flush balanced window start positions."""
    if total <= w:
        return [0]
    rem = total % w
    if rem == 0:
        return list(range(0, total, w))
    head = rem // 2
    s, pos = [0], w - head
    while pos + w < total:
        s.append(pos)
        pos += w
    s.append(total - w)
    return s


def _build_window_index(img_ids: torch.Tensor, window_size: int,
                        overlap_padding: int = 0,
                        return_centroids: bool = False):
    """Partition image token indices into full-sized spatial windows.

    Every window is exactly window_size x window_size tokens. When the latent
    grid is not evenly divisible, the last window along each axis is shifted
    left/up so it is flush with the edge and still full-sized, overlapping
    slightly with the previous window.
    """
    # Integer arithmetic: low-precision id dtypes (bf16) round positions above 256,
    # which drops edge tokens from every window and breaks uniform tiling.
    rows = img_ids[:, 1].float().round().long()
    cols = img_ids[:, 2].float().round().long()

    # Normalise to zero-based so window partitioning works regardless of
    # coordinate offset (when img_ids are shifted for text-distance capping).
    row_min = int(rows.min().item())
    col_min = int(cols.min().item())
    rows = rows - row_min
    cols = cols - col_min

    img_h = int(rows.max().item()) + 1
    img_w = int(cols.max().item()) + 1

    row_starts_list = _starts_axis(img_h, window_size)
    col_starts_list = _starts_axis(img_w, window_size)

    indices = []
    centroids = []
    for r in row_starts_list:
        for c in col_starts_list:
            r0 = max(0,     r - overlap_padding)
            r1 = min(img_h, r + window_size + overlap_padding)
            c0 = max(0,     c - overlap_padding)
            c1 = min(img_w, c + window_size + overlap_padding)
            mask = (rows >= r0) & (rows < r1) & \
                   (cols >= c0) & (cols < c1)
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                indices.append(idx)
                cr = (r0 + r1) // 2 + row_min
                cc = (c0 + c1) // 2 + col_min
                centroids.append((cr, cc))

    if return_centroids:
        return indices, centroids
    return indices


_ENTROPY_SAMPLE_CAP = 256  # max tokens to subsample — keeps matmul O(256^2)


@torch.no_grad()
def _window_entropy_coefficient(
    q_img: torch.Tensor,
    k_img: torch.Tensor,
    attn_scale_floor: float,
    head_dim: int,
) -> float:
    """Per-call softmax temperature coefficient based on attention entropy.

    Deterministic stride subsample (no randomness) up to _ENTROPY_SAMPLE_CAP.
    High entropy → coefficient approaches attn_scale_floor; low entropy →
    coefficient stays near 1.0.
    """
    if attn_scale_floor >= 1.0:
        return 1.0

    N_img = q_img.shape[1]
    stride = max(1, N_img // _ENTROPY_SAMPLE_CAP)
    idx = torch.arange(0, N_img, stride, device=q_img.device)[:_ENTROPY_SAMPLE_CAP]
    n = idx.shape[0]

    q = q_img[0, idx].float()
    k = k_img[0, idx].float()

    scale = 1.0 / math.sqrt(head_dim)
    logits = torch.bmm(
        q.permute(1, 0, 2),
        k.permute(1, 2, 0),
    ) * scale

    probs = torch.softmax(logits, dim=-1)
    mean_dist = probs.mean(dim=(0, 1))

    eps = 1e-9
    entropy = -(mean_dist * (mean_dist + eps).log()).sum()
    norm_entropy = (entropy / math.log(n)).clamp(0.0, 1.0).item()
    coefficient = attn_scale_floor + (1.0 - attn_scale_floor) * (1.0 - norm_entropy)
    return float(coefficient)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def _get_projections(attn, hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)
    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn, hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn, hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class FluxAttnProcessor:
    _attention_backend = None
    _viz_collector = None
    _viz_step = -1

    def __init__(self):
        pass

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Visualization: record sampled attention entropy/logit stats once per step
        coll = self.__class__._viz_collector
        step_idx = int(getattr(self.__class__, "_viz_step", -1))
        if coll is not None and step_idx >= 0 and getattr(coll, "_last_attn_step", None) != step_idx:
            try:
                B, S_joint, H, Dh = query.shape
                heads_sel = min(2, H)
                n_q = min(128, S_joint)
                # Keep the key softmax support fixed across resolutions where possible, so
                # entropy is measured over a comparable K.
                K = 512
                n_k = min(K, S_joint)

                q_idx = torch.linspace(0, max(S_joint - 1, 0), steps=n_q, device=query.device).long()
                k_idx = torch.linspace(0, max(S_joint - 1, 0), steps=n_k, device=key.device).long()

                q = query[:, q_idx, :heads_sel, :].float()
                k = key[:, k_idx, :heads_sel, :].float()

                logits = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(float(Dh))
                probs = torch.softmax(logits, dim=-1)
                eps = 1e-8
                entropy = (-(probs * (probs + eps).log()).sum(dim=-1)).mean().item()
                # Normalize entropy to [0, 1] under the assumption of a K-way distribution.
                # When S_joint < K, we fall back to the effective n_k.
                denom = math.log(max(int(n_k), 2))
                entropy_norm = (float(entropy) / float(denom)) if denom > 0 else None
                logit_std = logits.std().item()
                peakiness = (logits.max(dim=-1).values - logits.mean(dim=-1)).mean().item()
                max_prob = probs.max(dim=-1).values.mean().item()

                coll.record_attn(
                    step=step_idx,
                    entropy=float(entropy),
                    entropy_norm=(float(entropy_norm) if entropy_norm is not None else None),
                    k_used=int(n_k),
                    logit_std=float(logit_std),
                    peakiness=float(peakiness),
                    max_prob=float(max_prob),
                )

                # Visualization: save spatial attention maps for selected steps.
                # We compute attention from a small set of probe *image* query tokens to all keys,
                # then reshape the image-key part into [P, img_h, img_w].
                if step_idx in (5, 18, 27):
                    try:
                        txt_len = int(getattr(self.__class__, "_viz_txt_len", 0))
                        img_hw = getattr(self.__class__, "_viz_img_hw", None)
                        if img_hw is not None:
                            img_h, img_w = int(img_hw[0]), int(img_hw[1])
                            n_img = max(0, S_joint - txt_len)
                            if img_h * img_w == n_img and n_img > 0:
                                # 9 probes like smaple.py (top-left, top-center, ..., bot-right)
                                rel = [0.125, 0.5, 0.875]
                                probe_rc = [(r, c) for r in rel for c in rel]
                                probe_idx = []
                                for rr, cc in probe_rc:
                                    r = min(int(rr * img_h), img_h - 1)
                                    c = min(int(cc * img_w), img_w - 1)
                                    probe_idx.append(txt_len + r * img_w + c)

                                probe_q = query[:, probe_idx, :, :].float()  # [B,P,H,D]
                                scale = float(Dh) ** -0.5
                                scores = torch.einsum("bphd,bshd->bphs", probe_q, key.float()) * scale  # [B,P,H,S]
                                attn_w = scores.softmax(dim=-1).mean(dim=2)[0]  # [P,S]
                                img_attn = attn_w[:, txt_len:].reshape(len(probe_idx), img_h, img_w).detach().cpu().float()
                                # The image token grid in FLUX is vertically inverted vs the
                                # usual display convention; flip so maps align with rendered RGB.
                                img_attn = torch.flip(img_attn, dims=[1])
                                if not hasattr(coll, "_attn_maps"):
                                    coll._attn_maps = {}
                                if step_idx not in coll._attn_maps:
                                    coll._attn_maps[step_idx] = img_attn
                    except Exception:
                        pass
                coll._last_attn_step = step_idx
            except Exception:
                pass

        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, backend=self._attention_backend)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states


class FluxAttnProcessorChunked:
    """Chunked attention processor for HSWA phases 2 and 3.

    Replaces the single O(S^2) SDPA call with one small SDPA call per spatial
    window. Each call attends over [selected text tokens | window image tokens]
    only, skipping all out-of-window pairs entirely - no `-inf` mask needed.
    With ``window_text_min_frac``, the kept text count scales with window size
    (100%% at max schedule window; ``min_frac`` at floor); tokens are ranked by
    Q·K affinity to each spatial window.

    The new-sega FluxAttnProcessor handles the global / windowed-with-dense-mask
    paths; this processor is installed only when --optimized is set and a
    window is active.  RoPE is applied here exactly as in the new-sega
    processor (apply_rotary_emb on Q and K with the freqs produced by
    FluxPosEmbed), so sega mscale propagates transparently into the chunked
    path.
    """

    _attention_backend = None

    def __init__(
        self,
        txt_len: int,
        img_ids: torch.Tensor,
        window_size: int,
        overlap_padding: int = 0,
        attn_scale_floor: float = 1.0,
        local_text_pos: bool = False,
        pos_embed=None,
        ntk_factor: float = 1.0,
        rope_method: str = "ntk",
        axes_dim: Optional[List[int]] = None,
        per_window_sega: bool = False,
        txt_ids: Optional[torch.Tensor] = None,
        target_resolution: int = 1024,
        window_floor: int = 0,
        max_patches: int = 0,
        spread_min: float = 0.0,
        spread_max: float = 1.0,
        spread_alpha: float = 1.5,
        window_text_min_frac: Optional[float] = None,
        window_local_origin: bool = False,
        dwt_window_leaves: Optional[List[Any]] = None,
        leaf_ntk_factors: Optional[List[float]] = None,
        leaf_sega_blends: Optional[List[float]] = None,
        txt_groups: int = 1,
    ):
        self.txt_len = int(txt_len)
        self.txt_groups = max(1, int(txt_groups))
        self._dwt_mode = dwt_window_leaves is not None and len(dwt_window_leaves) > 0
        self._leaf_ntk = list(leaf_ntk_factors or [])
        self._leaf_sega_blend = list(leaf_sega_blends or [])
        self.window_local_origin = bool(window_local_origin)
        self.window_text_min_frac = (
            None if window_text_min_frac is None else float(window_text_min_frac)
        )
        self._effective_text_cap = compute_window_text_cap(
            window_size,
            window_floor,
            max_patches,
            txt_len,
            self.window_text_min_frac,
        )
        self.img_ids = img_ids
        self.window_size = window_size
        self.overlap_padding = overlap_padding
        self.attn_scale_floor = attn_scale_floor
        self.local_text_pos = local_text_pos
        self.pos_embed = pos_embed
        self.ntk_factor = ntk_factor
        self.rope_method = rope_method
        self.axes_dim = axes_dim
        self.per_window_sega = bool(per_window_sega)
        self.txt_ids = txt_ids
        self.target_resolution = int(target_resolution)
        self.window_floor = int(window_floor)
        self.max_patches = int(max_patches)
        self.spread_min = float(spread_min)
        self.spread_max = float(spread_max)
        self.spread_alpha = float(spread_alpha)

        if self._dwt_mode:
            from dwt_windows import leaves_to_window_indices

            self.window_indices = leaves_to_window_indices(
                img_ids, dwt_window_leaves, overlap_padding=overlap_padding,
            )
            self.window_centroids = None
            self._dwt_leaf_sizes = [int(lf.max_side) for lf in dwt_window_leaves]
        elif local_text_pos and pos_embed is not None:
            self.window_indices, self.window_centroids = _build_window_index(
                img_ids, window_size, overlap_padding=overlap_padding,
                return_centroids=True)
            self._dwt_leaf_sizes = None
        else:
            self.window_indices = _build_window_index(
                img_ids, window_size, overlap_padding=overlap_padding)
            self.window_centroids = None
            self._dwt_leaf_sizes = None

        # CPU indices: safe across multi-GPU block shards (moved to active device per forward).
        self.window_indices = [idx.cpu() for idx in self.window_indices]

        # Per-window text streams: the text sequence is W groups of txt_len/W tokens,
        # group i conditioning only window i (no cross-window text sharing).
        if self.txt_groups > 1:
            if self._dwt_mode:
                raise ValueError("txt_groups > 1 is incompatible with DWT quadtree windows")
            if self._effective_text_cap is not None:
                raise ValueError("txt_groups > 1 is incompatible with window_text_min_frac")
            if self.per_window_sega or self.local_text_pos:
                raise ValueError("txt_groups > 1 is incompatible with per-window SEGA / local_text_pos")
            if self.txt_len % self.txt_groups != 0:
                raise ValueError(
                    f"txt_len ({self.txt_len}) must be divisible by txt_groups ({self.txt_groups})",
                )
            if len(self.window_indices) != self.txt_groups:
                raise ValueError(
                    f"txt_groups ({self.txt_groups}) must equal the number of attention "
                    f"windows ({len(self.window_indices)})",
                )

        win_sizes = [idx.shape[0] for idx in self.window_indices]
        uniform_leaf = (
            not self._dwt_mode
            and len(set(win_sizes)) == 1
            and win_sizes
            and win_sizes[0] == window_size * window_size
        )
        self._use_batched = (
            not self._dwt_mode
            and self._effective_text_cap is None
            and uniform_leaf
            and len(self.window_indices) > 1
        )
        if self.per_window_sega:
            self._use_batched = False

        self._img_local_rope_idx = None
        if self.window_local_origin and uniform_leaf:
            self._img_local_rope_idx = _build_img_local_rope_index(img_ids, window_size).cpu()

        if self._use_batched:
            self._all_idx      = torch.stack(self.window_indices, dim=0)
            self._all_idx_flat = self._all_idx.view(-1)
            self._W            = win_sizes[0]
            self._img_count_cpu = None
        else:
            self._all_idx = self._all_idx_flat = self._W = self._img_count_cpu = None

    def _apply_window_sega_rope(
        self,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        win_idx: torch.Tensor,
        hidden_states: torch.Tensor,
        is_dual_stream: bool,
        txt_row_indices: Optional[torch.Tensor] = None,
        leaf_i: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE from SEGA spectra on the spatial crop for ``win_idx`` (chunked HSWA only)."""
        if is_dual_stream:
            img_hidden = hidden_states
        else:
            img_hidden = hidden_states[:, self.txt_len :, :]
        crop, h, w = _crop_window_spatial_hidden(img_hidden, self.img_ids, win_idx)
        crop_seq = crop.reshape(crop.shape[0], h * w, -1)
        n_bins_h = max(h // 2, 1)
        n_bins_w = max(w // 2, 1)
        n_bins_iso = max(max(h, w) // 2, 1)
        eh, ew = compute_axis_spectral_profiles(crop_seq, h, w, n_bins_h, n_bins_w)
        iso = compute_spectral_energy_profile(crop_seq, h, w, n_bins_iso)
        spread = compute_dynamic_spread(
            iso, spread_min=self.spread_min,
            spread_max=self.spread_max, alpha=self.spread_alpha,
        )
        ws_eff = (
            int(self._dwt_leaf_sizes[leaf_i])
            if self._dwt_mode and self._dwt_leaf_sizes and leaf_i < len(self._dwt_leaf_sizes)
            else int(self.window_size)
        )
        lam = self._sega_blend_for_leaf(leaf_i, ws_eff)
        dev = q_w.device
        txt_ids_full = self.txt_ids.to(device=dev, dtype=self.img_ids.dtype)
        if txt_row_indices is not None:
            txt_ids = txt_ids_full[txt_row_indices]
        else:
            txt_ids = txt_ids_full
        ids_w = torch.cat([txt_ids, self.img_ids.to(device=dev)[win_idx]], dim=0)
        pe = self.pos_embed
        old_viz = getattr(pe, "_viz_collector", None)
        pe._viz_collector = None
        try:
            cos_w, sin_w = pe(
                ids_w,
                ntk_factor=float(self._ntk_for_leaf(leaf_i)),
                mscale_spread=spread,
                energy_profile_h=eh,
                energy_profile_w=ew,
                target_res=int(self.target_resolution),
                sega_excess_blend=float(lam),
            )
        finally:
            pe._viz_collector = old_viz
        q_wr = apply_rotary_emb(q_w, (cos_w, sin_w), sequence_dim=1)
        k_wr = apply_rotary_emb(k_w, (cos_w, sin_w), sequence_dim=1)
        return q_wr, k_wr

    def _leaf_index_for_win_idx(self, win_idx: torch.Tensor) -> int:
        """Map ``win_idx`` tensor to leaf index (DWT mode)."""
        if not self._dwt_mode:
            return 0
        key = int(win_idx[0].item()) if win_idx.numel() else 0
        for i, widx in enumerate(self.window_indices):
            if widx.numel() and int(widx[0].item()) == key:
                return i
        return 0

    def _leaf_size_for_index(self, win_idx: torch.Tensor) -> int:
        if self._dwt_mode and self._dwt_leaf_sizes:
            li = self._leaf_index_for_win_idx(win_idx)
            if li < len(self._dwt_leaf_sizes):
                return int(self._dwt_leaf_sizes[li])
        return int(self.window_size)

    def _ntk_for_leaf(self, leaf_i: int) -> float:
        if self._dwt_mode and self._leaf_ntk and leaf_i < len(self._leaf_ntk):
            return float(self._leaf_ntk[leaf_i])
        return float(self.ntk_factor)

    def _sega_blend_for_leaf(self, leaf_i: int, ws_eff: int) -> float:
        if self._dwt_mode and self._leaf_sega_blend and leaf_i < len(self._leaf_sega_blend):
            return float(self._leaf_sega_blend[leaf_i])
        return float(compute_sega_excess_blend(
            ws_eff, self.window_floor, self.max_patches, rope_method="sega",
        ))

    def _window_local_rope_emb(
        self,
        win_idx: torch.Tensor,
        n_text: int,
        hidden_states: torch.Tensor,
        is_dual_stream: bool,
        leaf_i: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cos/sin for window-local RoPE: text at (0,0), image patches from window origin."""
        if self.pos_embed is None or self.txt_ids is None:
            raise RuntimeError("window_local_origin requires pos_embed and txt_ids")
        return compute_window_local_rope_emb(
            pos_embed=self.pos_embed,
            img_ids=self.img_ids,
            win_idx=win_idx,
            n_text=n_text,
            window_size=self.window_size,
            rope_method=self.rope_method,
            spread_min=self.spread_min,
            spread_max=self.spread_max,
            spread_alpha=self.spread_alpha,
            hidden_states=hidden_states,
            txt_len=self.txt_len,
            is_dual_stream=is_dual_stream,
            leaf_i=leaf_i,
            dwt_leaf_sizes=self._dwt_leaf_sizes,
            leaf_ntk_factors=self._leaf_ntk,
            leaf_sega_blends=self._leaf_sega_blend,
            window_floor=self.window_floor,
            max_patches=self.max_patches,
        )

    def _apply_cached_uniform_local_rope(
        self,
        q_txt: torch.Tensor,
        k_txt: torch.Tensor,
        q_img: torch.Tensor,
        k_img: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        txt_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rotate text + image Q/K once over the full sequence (same cost as global RoPE)."""
        cos, sin = image_rotary_emb
        cos_t, sin_t = cos[:txt_len], sin[:txt_len]
        q_txt = apply_rotary_emb(q_txt, (cos_t, sin_t), sequence_dim=1)
        k_txt = apply_rotary_emb(k_txt, (cos_t, sin_t), sequence_dim=1)
        idx = self._img_local_rope_idx.to(device=device)
        cos_img = cos[txt_len:][idx]
        sin_img = sin[txt_len:][idx]
        q_img = apply_rotary_emb(q_img, (cos_img, sin_img), sequence_dim=1)
        k_img = apply_rotary_emb(k_img, (cos_img, sin_img), sequence_dim=1)
        return q_txt, k_txt, q_img, k_img

    def _apply_window_local_origin_rope(
        self,
        q_txt: torch.Tensor,
        k_txt: torch.Tensor,
        q_img_w: torch.Tensor,
        k_img_w: torch.Tensor,
        win_idx: torch.Tensor,
        hidden_states: torch.Tensor,
        is_dual_stream: bool,
        leaf_i: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE as if the window were a standalone image: text at (0,0), patches from origin."""
        cos_w, sin_w = self._window_local_rope_emb(
            win_idx, int(q_txt.shape[1]), hidden_states, is_dual_stream, leaf_i=leaf_i,
        )
        q_w = torch.cat([q_txt, q_img_w], dim=1)
        k_w = torch.cat([k_txt, k_img_w], dim=1)
        q_wr = apply_rotary_emb(q_w, (cos_w, sin_w), sequence_dim=1)
        k_wr = apply_rotary_emb(k_w, (cos_w, sin_w), sequence_dim=1)
        return q_wr, k_wr

    def _chunked_text_slices(
        self,
        q_txt: torch.Tensor,
        k_txt: torch.Tensor,
        v_txt: torch.Tensor,
        q_img_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        """Subset text Q/K/V when ``_effective_text_cap`` is set for this window size."""
        cap = self._effective_text_cap
        if cap is None:
            return q_txt, k_txt, v_txt, self.txt_len, None
        txt_idx = _select_window_text_indices(q_img_w, k_txt, cap)
        if txt_idx is None:
            return q_txt, k_txt, v_txt, self.txt_len, None
        return (
            _gather_seq_dim1(q_txt, txt_idx),
            _gather_seq_dim1(k_txt, txt_idx),
            _gather_seq_dim1(v_txt, txt_idx),
            int(txt_idx.shape[1]),
            txt_idx,
        )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,  # ignored
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key   = key.unflatten(-1,   (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key   = attn.norm_k(key)

        is_dual_stream = attn.added_kv_proj_dim is not None
        if is_dual_stream:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key   = encoder_key.unflatten(-1,   (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key   = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key   = torch.cat([encoder_key,   key],   dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        defer_window_sega_rope = (
            self.per_window_sega
            and self.rope_method == "sega"
            and self.pos_embed is not None
            and self.txt_ids is not None
        )
        if (
            image_rotary_emb is not None
            and not defer_window_sega_rope
            and not self.window_local_origin
        ):
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key   = apply_rotary_emb(key,   image_rotary_emb, sequence_dim=1)

        B, S, H, D = query.shape
        txt_len = self.txt_len

        q_txt = query[:, :txt_len]
        k_txt = key[:,   :txt_len]
        v_txt = value[:, :txt_len]

        # window_local_origin skips global RoPE above; Q/K are unrotated here.
        q_img = query[:, txt_len:]
        k_img = key[:, txt_len:]
        v_img = value[:, txt_len:]
        dev = query.device

        applied_uniform_local = False
        if (
            self.window_local_origin
            and image_rotary_emb is not None
            and self._img_local_rope_idx is not None
        ):
            q_txt, k_txt, q_img, k_img = self._apply_cached_uniform_local_rope(
                q_txt, k_txt, q_img, k_img, image_rotary_emb, txt_len, dev,
            )
            applied_uniform_local = True

        if self.local_text_pos and image_rotary_emb is not None and self.pos_embed is not None:
            cos_orig, sin_orig = image_rotary_emb
            txt_cos = cos_orig[:txt_len]
            txt_sin = sin_orig[:txt_len]
            q_txt_raw = apply_rotary_emb(q_txt, (txt_cos, -txt_sin), sequence_dim=1)
            k_txt_raw = apply_rotary_emb(k_txt, (txt_cos, -txt_sin), sequence_dim=1)
        else:
            q_txt_raw = k_txt_raw = None

        N_img  = q_img.shape[1]
        n_wins = len(self.window_indices)
        # Text tokens seen by one window: the full stream, or this window's group.
        t_per = txt_len // self.txt_groups

        if self.attn_scale_floor < 1.0:
            _txt_for_scale = (
                self._effective_text_cap if self._effective_text_cap is not None else t_per
            )
            _ws_scale = int(self.window_size)
            if self._dwt_mode and self._dwt_leaf_sizes:
                _ws_scale = max(self._dwt_leaf_sizes)
            _window_tokens = _txt_for_scale + _ws_scale * _ws_scale
            _total_tokens  = txt_len + N_img
            _base_coeff    = math.sqrt(_window_tokens / max(1, _total_tokens))
            _coeff         = max(self.attn_scale_floor, min(1.0, _base_coeff))
            sdpa_scale     = _coeff / math.sqrt(D)
        else:
            sdpa_scale = None

        if self._use_batched:
            W = self._W
            all_idx_flat = self._all_idx_flat.to(dev)

            q_wins = q_img[:, all_idx_flat].view(B, n_wins, W, H, D)
            k_wins = k_img[:, all_idx_flat].view(B, n_wins, W, H, D)
            v_wins = v_img[:, all_idx_flat].view(B, n_wins, W, H, D)

            if self.txt_groups > 1:
                # Per-window text: group i pairs with window i (no broadcast).
                q_txt_r = q_txt.reshape(B, n_wins, t_per, H, D)
                k_txt_r = k_txt.reshape(B, n_wins, t_per, H, D)
                v_txt_r = v_txt.reshape(B, n_wins, t_per, H, D)
            else:
                q_txt_r = q_txt.unsqueeze(1).expand(B, n_wins, t_per, H, D)
                k_txt_r = k_txt.unsqueeze(1).expand(B, n_wins, t_per, H, D)
                v_txt_r = v_txt.unsqueeze(1).expand(B, n_wins, t_per, H, D)

            q_b = torch.cat([q_txt_r, q_wins], dim=2).view(B * n_wins, t_per + W, H, D)
            k_b = torch.cat([k_txt_r, k_wins], dim=2).view(B * n_wins, t_per + W, H, D)
            v_b = torch.cat([v_txt_r, v_wins], dim=2).view(B * n_wins, t_per + W, H, D)

            if self.window_local_origin and not applied_uniform_local:
                seq_len = int(q_b.shape[1])
                if image_rotary_emb is not None and image_rotary_emb[0].shape[0] == seq_len:
                    cos_w, sin_w = image_rotary_emb
                else:
                    cos_w, sin_w = self._window_local_rope_emb(
                        self.window_indices[0].to(device=dev),
                        t_per,
                        hidden_states,
                        is_dual_stream,
                    )
                q_b = apply_rotary_emb(q_b, (cos_w, sin_w), sequence_dim=1)
                k_b = apply_rotary_emb(k_b, (cos_w, sin_w), sequence_dim=1)

            out_b = dispatch_attention_fn(
                q_b, k_b, v_b, attn_mask=None,
                scale=sdpa_scale,
                backend=self._attention_backend,
            )

            out_b = out_b.view(B, n_wins, t_per + W, H, D)

            if self._img_count_cpu is None:
                ic = torch.zeros(N_img, dtype=torch.float32)
                ic.scatter_add_(0, all_idx_flat.cpu(),
                                torch.ones(n_wins * W, dtype=torch.float32))
                self._img_count_cpu = ic
            img_count = self._img_count_cpu.to(device=dev, dtype=query.dtype)

            out_img = torch.zeros_like(q_img)
            img_out_flat = out_b[:, :, t_per:].reshape(B, n_wins * W, H, D)
            idx_expand   = all_idx_flat.view(1, -1, 1, 1).expand(B, -1, H, D)
            out_img.scatter_add_(1, idx_expand, img_out_flat)
            out_img = out_img / img_count.clamp(min=1.0).view(1, N_img, 1, 1)

            if self.txt_groups > 1:
                # Each text group belongs to exactly one window: concatenate back.
                out_txt = out_b[:, :, :t_per].reshape(B, txt_len, H, D)
            else:
                out_txt = out_b[:, :, :t_per].mean(dim=1)

        else:
            out_img    = torch.zeros_like(q_img)
            img_count  = torch.zeros(N_img, dtype=query.dtype, device=query.device)
            out_txt    = torch.zeros(B, txt_len, H, D,
                                     dtype=query.dtype, device=query.device)
            txt_weight = torch.zeros(B, txt_len, dtype=query.dtype, device=query.device)

            for wi, win_idx in enumerate(self.window_indices):
                win_idx = win_idx.to(device=dev)
                q_img_w = q_img[:, win_idx]
                if self.txt_groups > 1:
                    tsl = slice(wi * t_per, (wi + 1) * t_per)
                    q_t, k_t, v_t = q_txt[:, tsl], k_txt[:, tsl], v_txt[:, tsl]
                    t_eff, txt_idx = t_per, None
                else:
                    tsl = None
                    q_t, k_t, v_t, t_eff, txt_idx = self._chunked_text_slices(
                        q_txt, k_txt, v_txt, q_img_w,
                    )
                v_w = torch.cat([v_t, v_img[:, win_idx]], dim=1)
                if self.window_local_origin:
                    if applied_uniform_local:
                        q_w = torch.cat([q_t, q_img_w], dim=1)
                        k_w = torch.cat([k_t, k_img[:, win_idx]], dim=1)
                    elif (
                        image_rotary_emb is not None
                        and image_rotary_emb[0].shape[0] == t_eff + win_idx.shape[0]
                    ):
                        cos_w, sin_w = image_rotary_emb
                        q_w = torch.cat([q_t, q_img_w], dim=1)
                        k_w = torch.cat([k_t, k_img[:, win_idx]], dim=1)
                        q_w = apply_rotary_emb(q_w, (cos_w, sin_w), sequence_dim=1)
                        k_w = apply_rotary_emb(k_w, (cos_w, sin_w), sequence_dim=1)
                    else:
                        q_w, k_w = self._apply_window_local_origin_rope(
                            q_t, k_t, q_img_w, k_img[:, win_idx],
                            win_idx, hidden_states, is_dual_stream,
                            leaf_i=wi,
                        )
                else:
                    q_w = torch.cat([q_t, q_img_w], dim=1)
                    k_w = torch.cat([k_t, k_img[:, win_idx]], dim=1)
                    txt_rows = None
                    if txt_idx is not None:
                        txt_rows = txt_idx[0] if B == 1 else None
                    if defer_window_sega_rope:
                        q_w, k_w = self._apply_window_sega_rope(
                            q_w, k_w, win_idx, hidden_states, is_dual_stream,
                            txt_row_indices=txt_rows,
                            leaf_i=wi,
                        )

                out_w = dispatch_attention_fn(
                    q_w, k_w, v_w, attn_mask=None,
                    scale=sdpa_scale,
                    backend=self._attention_backend,
                )

                out_img[:, win_idx] += out_w[:, t_eff:]
                img_count[win_idx]  += 1.0
                if tsl is not None:
                    out_txt[:, tsl] = out_txt[:, tsl] + out_w[:, :t_eff]
                    txt_weight[:, tsl] = txt_weight[:, tsl] + 1.0
                elif txt_idx is None:
                    out_txt = out_txt + out_w[:, :t_eff]
                    txt_weight = txt_weight + 1.0
                else:
                    idx_scatter = (
                        txt_idx.unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand(B, t_eff, H, D)
                    )
                    out_txt.scatter_add_(1, idx_scatter, out_w[:, :t_eff])
                    txt_weight.scatter_add_(
                        1, txt_idx, torch.ones(B, t_eff, device=dev, dtype=query.dtype),
                    )

            out_img = out_img / img_count.clamp(min=1.0).view(1, N_img, 1, 1)
            denom = txt_weight.clamp(min=1.0).unsqueeze(-1).unsqueeze(-1)
            out_txt = out_txt / denom

        hidden_states = torch.cat([out_txt, out_img], dim=1)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if is_dual_stream:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [txt_len, hidden_states.shape[1] - txt_len], dim=1
            )
            hidden_states         = attn.to_out[0](hidden_states)
            hidden_states         = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = FluxAttnProcessor
    _available_processors = [FluxAttnProcessor, FluxAttnProcessorChunked]

    def __init__(
        self, query_dim, heads=8, dim_head=64, dropout=0.0, bias=False,
        added_kv_proj_dim=None, added_proj_bias=True, out_bias=True,
        eps=1e-5, out_dim=None, context_pre_only=None, pre_only=False,
        elementwise_affine=True, processor=None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = torch.nn.ModuleList([
                torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
                torch.nn.Dropout(dropout),
            ])

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kwargs):
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
        self.attn = FluxAttention(
            query_dim=dim, dim_head=attention_head_dim, heads=num_attention_heads,
            out_dim=dim, bias=True, processor=FluxAttnProcessor(), eps=1e-6, pre_only=True,
        )

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb, **(joint_attention_kwargs or {}))
        hidden_states = gate.unsqueeze(1) * self.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.attn = FluxAttention(
            query_dim=dim, added_kv_proj_dim=dim, dim_head=attention_head_dim,
            heads=num_attention_heads, out_dim=dim, context_pre_only=False,
            bias=True, processor=FluxAttnProcessor(), eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None, attention_mask=None):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb, **(joint_attention_kwargs or {}),
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hidden_states)
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_encoder_hidden_states)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


# ---------------------------------------------------------------------------
# Position embedding
# ---------------------------------------------------------------------------

class FluxPosEmbed(nn.Module):
    def __init__(self, theta, axes_dim, training_resolution=64,
                 mscale_alpha=0.15, mscale_beta=1.5, mscale_min=1.0,
                 base_mscale_formula="power_res", base_mscale_coefficient=None,
                 training_res_pixels=None):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.training_resolution = training_resolution
        self.mscale_alpha = mscale_alpha
        self.mscale_beta = mscale_beta
        self.mscale_min = mscale_min
        self.base_mscale_formula = base_mscale_formula
        self.base_mscale_coefficient = base_mscale_coefficient
        self.training_res_pixels = training_res_pixels

    def forward(
        self,
        ids,
        ntk_factor=1.0,
        mscale_spread=None,
        energy_profile_h=None,
        energy_profile_w=None,
        target_res=None,
        sega_excess_blend: float = 1.0,
        sega_base_mscale_res_ratio: Optional[float] = None,
    ):
        n_axes = ids.shape[-1]
        cos_out, sin_out = [], []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        axis_energy = {1: energy_profile_h, 2: energy_profile_w}
        disable_mscale = bool(getattr(self, "_disable_mscale", False))
        use_yarn = bool(getattr(self, "_use_yarn", False))

        for i in range(n_axes):
            if i > 0 and use_yarn:
                current_patches = int(pos[:, i].max().item()) + 1
                cos, sin = get_1d_rotary_pos_embed(
                    self.axes_dim[i], pos[:, i], theta=self.theta,
                    repeat_interleave_real=True, use_real=True, freqs_dtype=freqs_dtype,
                    yarn=True,
                    max_pe_len=current_patches,
                    ori_max_pe_len=int(self.training_resolution),
                )
            elif i > 0 and ntk_factor > 1.0:
                cos, sin = get_1d_rotary_pos_embed(
                    self.axes_dim[i], pos[:, i], theta=self.theta,
                    repeat_interleave_real=True, use_real=True, freqs_dtype=freqs_dtype,
                    ntk_factor=ntk_factor,
                    energy_profile=axis_energy.get(i),
                    mscale_spread=mscale_spread,
                    mscale_alpha=self.mscale_alpha, mscale_beta=self.mscale_beta,
                    mscale_min=self.mscale_min,
                    base_mscale_formula=self.base_mscale_formula,
                    base_mscale_coefficient=self.base_mscale_coefficient,
                    target_res=target_res, training_res=self.training_res_pixels,
                    viz_collector=getattr(self, "_viz_collector", None),
                    viz_step=int(getattr(self, "_viz_step", -1)),
                    viz_axis=("height" if i == 1 else "width"),
                    disable_mscale=disable_mscale,
                    sega_excess_blend=float(sega_excess_blend),
                    sega_base_mscale_res_ratio=sega_base_mscale_res_ratio,
                )
            else:
                cos, sin = get_1d_rotary_pos_embed(
                    self.axes_dim[i], pos[:, i], theta=self.theta,
                    repeat_interleave_real=True, use_real=True, freqs_dtype=freqs_dtype,
                    sega_base_mscale_res_ratio=sega_base_mscale_res_ratio,
                )
            cos_out.append(cos)
            sin_out.append(sin)

        return torch.cat(cos_out, dim=-1).to(ids.device), torch.cat(sin_out, dim=-1).to(ids.device)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class Flux1Transformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin, CacheMixin, AttentionMixin,
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        training_resolution: int = 1024,
        spread_max: float = 1.0,
        spread_min: float = 0.0,
        spread_alpha: float = 1.5,
        mscale_alpha: float = 0.15,
        mscale_beta: float = 1.5,
        mscale_min: float = 1.0,
        base_mscale_formula: str = "power_res",
        base_mscale_coefficient: float = 0.08,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        training_resolution_tokens = training_resolution / 16

        self.spread_min = spread_min
        self.spread_max = spread_max
        self.spread_alpha = spread_alpha
        self.training_resolution = training_resolution
        self._rope_spatial_dim = axes_dims_rope[1]

        self.pos_embed = FluxPosEmbed(
            theta=10000, axes_dim=axes_dims_rope,
            training_resolution=training_resolution_tokens,
            mscale_alpha=mscale_alpha, mscale_beta=mscale_beta, mscale_min=mscale_min,
            base_mscale_formula=base_mscale_formula,
            base_mscale_coefficient=base_mscale_coefficient,
            training_res_pixels=training_resolution,
        )

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(dim=self.inner_dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim)
            for _ in range(num_layers)
        ])
        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(dim=self.inner_dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim)
            for _ in range(num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.gradient_checkpointing = False
        # HSWA: track which processor flavour is currently installed so we
        # don't reinstall it every step.
        self._chunked_proc_active = False
        self._chunked_window = -1
        self._chunked_text_min_frac: Optional[float] = None
        self._chunked_window_local_origin: bool = False
        self._chunked_txt_groups: int = 1
        self._chunked_txt_len: int = -1
        self._hswa_dwt_windows: bool = False
        self._hswa_dwt_levels: int = 3
        self._hswa_dwt_tau: Optional[float] = None
        self._hswa_dwt_tau_percentile: float = 40.0
        self._hswa_dwt_micro_tile: int = 8
        self._hswa_dwt_high_pass: bool = False
        self._hswa_dwt_quadtree: bool = False
        self._hswa_dwt_hybrid: bool = False
        self._hswa_dwt_radial: bool = False
        self._hswa_dwt_fine_size: Optional[int] = None
        self._hswa_dwt_terrain_tiers: int = 5
        self._hswa_dwt_variance_gate: bool = True
        self._hswa_dwt_variance_min: float = 1e-6
        self._hswa_dwt_spectrum_min_fraction: float = 0.2
        self._hswa_dwt_max_windows: Optional[int] = None
        self._hswa_dwt_max_windows_frac: float = 1.0
        self._hswa_dwt_spectrum_smooth: int = 5
        self._hswa_dwt_terrain_inner_radius: Optional[float] = None
        self._hswa_dwt_terrain_outer_radius: Optional[float] = None
        self._hswa_dwt_terrain_transition_pad: Optional[int] = None
        self._hswa_dwt_terrain_detail_tau: Optional[float] = None
        self._hswa_dwt_terrain_detail_percentile: float = 75.0
        self._hswa_dwt_terrain_max_peaks: Optional[int] = None
        self._hswa_dwt_run_seed: int = 0
        self._hswa_dwt_fine_stochastic: bool = True
        self._hswa_dwt_fine_score_noise: float = 0.25
        self._hswa_dwt_fine_jitter: int = 16
        self._hswa_dwt_fine_top_frac: float = 0.12
        self._hswa_overlap_padding: int = 0
        self._dwt_plot_prefix: Optional[str] = None
        self._dwt_saved_steps: Optional[set] = None

    # ── HSWA processor management ────────────────────────────────────────────

    @torch.no_grad()
    def _compute_hswa_dwt_leaves(
        self,
        hidden_states: torch.Tensor,
        *,
        img_h: int,
        img_w: int,
        window_size: int,
        window_floor: int,
        max_patches: int,
        txt_len: int,
        rope_method: str,
        img_ids: Optional[torch.Tensor] = None,
    ):
        from dwt_windows import (
            attach_leaf_rope_params,
            build_hybrid_dwt_windows,
            build_quadtree_windows,
            build_spectrum_dwt_windows,
            build_terrain_dwt_windows,
            detail_energy_map,
            hidden_to_scalar_map,
            save_dwt_plots,
        )

        sm = hidden_to_scalar_map(
            hidden_states,
            img_h,
            img_w,
            high_pass=bool(self._hswa_dwt_high_pass),
            img_ids=img_ids,
        )
        energy = detail_energy_map(
            sm,
            levels=int(self._hswa_dwt_levels),
            micro_tile=int(self._hswa_dwt_micro_tile),
        )
        overlap_pad = int(getattr(self, "_hswa_overlap_padding", 0))
        if bool(getattr(self, "_hswa_dwt_quadtree", False)):
            leaves, qt_meta = build_quadtree_windows(
                energy,
                img_h=img_h,
                img_w=img_w,
                window_floor=int(window_floor),
                tau=self._hswa_dwt_tau,
                tau_percentile=float(self._hswa_dwt_tau_percentile),
                micro_tile=int(self._hswa_dwt_micro_tile),
                txt_len=int(txt_len),
                uniform_window_size=int(window_size),
            )
        elif bool(getattr(self, "_hswa_dwt_hybrid", False)):
            leaves, qt_meta = build_hybrid_dwt_windows(
                energy,
                img_h=img_h,
                img_w=img_w,
                window_size=int(window_size),
                window_floor=int(window_floor),
                overlap_padding=overlap_pad,
                micro_tile=int(self._hswa_dwt_micro_tile),
                fine_size_patches=getattr(self, "_hswa_dwt_fine_size", None),
                run_seed=int(getattr(self, "_hswa_dwt_run_seed", 0)),
                step=int(getattr(self, "_viz_step", 0)),
                fine_stochastic=bool(getattr(self, "_hswa_dwt_fine_stochastic", True)),
                fine_score_noise_scale=float(
                    getattr(self, "_hswa_dwt_fine_score_noise", 0.25),
                ),
                fine_position_jitter=int(getattr(self, "_hswa_dwt_fine_jitter", 16)),
                fine_top_frac=float(getattr(self, "_hswa_dwt_fine_top_frac", 0.12)),
            )
        elif bool(getattr(self, "_hswa_dwt_radial", False)):
            leaves, qt_meta = build_terrain_dwt_windows(
                energy,
                img_h=img_h,
                img_w=img_w,
                window_size=int(window_size),
                window_floor=int(window_floor),
                overlap_padding=overlap_pad,
                micro_tile=int(self._hswa_dwt_micro_tile),
                fine_size_patches=getattr(self, "_hswa_dwt_fine_size", None),
                run_seed=int(getattr(self, "_hswa_dwt_run_seed", 0)),
                step=int(getattr(self, "_viz_step", 0)),
                fine_stochastic=bool(getattr(self, "_hswa_dwt_fine_stochastic", True)),
                fine_score_noise_scale=float(
                    getattr(self, "_hswa_dwt_fine_score_noise", 0.25),
                ),
                fine_position_jitter=int(getattr(self, "_hswa_dwt_fine_jitter", 16)),
                fine_top_frac=float(getattr(self, "_hswa_dwt_fine_top_frac", 0.12)),
                terrain_tiers=int(getattr(self, "_hswa_dwt_terrain_tiers", 5)),
                terrain_inner_radius=getattr(self, "_hswa_dwt_terrain_inner_radius", None),
                terrain_outer_radius=getattr(self, "_hswa_dwt_terrain_outer_radius", None),
                terrain_transition_pad=getattr(self, "_hswa_dwt_terrain_transition_pad", None),
                terrain_detail_tau=getattr(self, "_hswa_dwt_terrain_detail_tau", None),
                terrain_detail_percentile=float(
                    getattr(self, "_hswa_dwt_terrain_detail_percentile", 75.0),
                ),
                terrain_max_peaks=getattr(self, "_hswa_dwt_terrain_max_peaks", None),
            )
        else:
            leaves, qt_meta = build_spectrum_dwt_windows(
                energy,
                img_h=img_h,
                img_w=img_w,
                window_size=int(window_size),
                window_floor=int(window_floor),
                overlap_padding=overlap_pad,
                micro_tile=int(self._hswa_dwt_micro_tile),
                spectrum_tiers=int(getattr(self, "_hswa_dwt_terrain_tiers", 5)),
                variance_gate=bool(getattr(self, "_hswa_dwt_variance_gate", True)),
                variance_min=float(getattr(self, "_hswa_dwt_variance_min", 1e-6)),
                min_tier_fraction=float(
                    getattr(self, "_hswa_dwt_spectrum_min_fraction", 0.2),
                ),
                max_windows=getattr(self, "_hswa_dwt_max_windows", None),
                max_windows_frac=float(getattr(self, "_hswa_dwt_max_windows_frac", 1.0)),
                spectrum_smooth=int(getattr(self, "_hswa_dwt_spectrum_smooth", 5)),
            )
        if not bool(getattr(self, "_hswa_dwt_global_sega_fft", False)):
            ntk_fn = getattr(self, "_hswa_dwt_ntk_fn", None) or _ntk_factor_for_window_patches
            attach_leaf_rope_params(
                leaves,
                window_floor=int(window_floor),
                max_patches=int(max_patches),
                rope_method=str(rope_method),
                ntk_for_window_fn=ntk_fn,
                sega_excess_blend_fn=compute_sega_excess_blend,
            )
        prefix = self._dwt_plot_prefix
        if prefix:
            step = int(getattr(self, "_viz_step", -1))
            saved = self._dwt_saved_steps
            if saved is None or step not in saved:
                save_dwt_plots(
                    prefix,
                    energy,
                    leaves,
                    img_h=img_h,
                    img_w=img_w,
                    step=step,
                    window_size=int(window_size),
                    quadtree_meta=qt_meta,
                )
                if saved is not None:
                    saved.add(step)
        return leaves, energy

    def _install_chunked_processors(
        self,
        txt_len: int,
        img_ids: torch.Tensor,
        window_size: int,
        overlap_padding: int = 0,
        attn_scale_floor: float = 1.0,
        local_text_pos: bool = False,
        ntk_factor: float = 1.0,
        rope_method: str = "ntk",
        debug_windows: bool = False,
        per_window_sega: bool = False,
        txt_ids: Optional[torch.Tensor] = None,
        target_resolution: int = 1024,
        window_floor: int = 0,
        max_patches: int = 0,
        spread_min: float = 0.0,
        spread_max: float = 1.0,
        spread_alpha: float = 1.5,
        window_text_min_frac: Optional[float] = None,
        window_local_origin: bool = False,
        dwt_window_leaves: Optional[List[Any]] = None,
        txt_groups: int = 1,
    ):
        """Swap all attention blocks to FluxAttnProcessorChunked."""
        if per_window_sega and txt_ids is None:
            raise ValueError("per_window_sega requires txt_ids")
        if window_local_origin and txt_ids is None:
            raise ValueError("window_local_origin requires txt_ids")
        if dwt_window_leaves and window_text_min_frac is not None:
            raise ValueError("DWT quadtree windows are incompatible with --hswa-window-text-min-frac")
        leaf_ntk = [float(lf.ntk_factor) for lf in dwt_window_leaves] if dwt_window_leaves else None
        leaf_blend = [float(lf.sega_excess_blend) for lf in dwt_window_leaves] if dwt_window_leaves else None
        kwargs = dict(
            txt_len=txt_len, img_ids=img_ids,
            window_size=window_size, overlap_padding=overlap_padding,
            attn_scale_floor=attn_scale_floor, local_text_pos=local_text_pos,
            pos_embed=self.pos_embed if (
                local_text_pos or per_window_sega or window_local_origin
            ) else None,
            ntk_factor=ntk_factor, rope_method=rope_method,
            axes_dim=self.pos_embed.axes_dim,
            per_window_sega=per_window_sega,
            txt_ids=txt_ids,
            target_resolution=target_resolution,
            window_floor=window_floor,
            max_patches=max_patches,
            spread_min=spread_min,
            spread_max=spread_max,
            spread_alpha=spread_alpha,
            window_text_min_frac=window_text_min_frac,
            window_local_origin=window_local_origin,
            dwt_window_leaves=dwt_window_leaves,
            leaf_ntk_factors=leaf_ntk,
            leaf_sega_blends=leaf_blend,
            txt_groups=txt_groups,
        )

        if debug_windows and not dwt_window_leaves:
            _rows = img_ids[:, 1] - img_ids[:, 1].min()
            _cols = img_ids[:, 2] - img_ids[:, 2].min()
            _img_h = int(_rows.max().item()) + 1
            _img_w  = int(_cols.max().item()) + 1
            _N_img  = _img_h * _img_w
            _rs = _starts_axis(_img_h, window_size)
            _cs = _starts_axis(_img_w, window_size)
            _n_wins = len(_rs) * len(_cs)
            _win_tokens = window_size * window_size
            _win_seq  = txt_len + _win_tokens
            _glob_seq = txt_len + _N_img
            _win_cost  = _n_wins * _win_seq ** 2
            _glob_cost = _glob_seq ** 2
            _speedup   = _glob_cost / max(1, _win_cost)
            print(f"\n[HSWA] ws={window_size}  grid={len(_rs)}x{len(_cs)}  n_wins={_n_wins}"
                  f"  SDPA/win: ({txt_len}+{_win_tokens})^2={_win_seq**2:,}"
                  f"  total={_win_cost:,}  global={_glob_cost:,}"
                  f"  speedup={_speedup:.2f}x")
        elif debug_windows and dwt_window_leaves:
            gfft = " +global-SEGA-FFT" if per_window_sega is False else " +per-leaf-SEGA"
            n_fine = sum(1 for lf in dwt_window_leaves if lf.is_fine)
            n_base = len(dwt_window_leaves) - n_fine
            print(
                f"\n[HSWA-DWT] sched_ws={window_size}  n_base={n_base}  n_fine={n_fine}"
                f"  (total={len(dwt_window_leaves)}){gfft}",
                flush=True,
            )

        for block in self.transformer_blocks:
            block.attn.set_processor(FluxAttnProcessorChunked(**kwargs))
        for block in self.single_transformer_blocks:
            block.attn.set_processor(FluxAttnProcessorChunked(**kwargs))
        self._chunked_proc_active = True
        self._chunked_window = window_size
        self._chunked_text_min_frac = window_text_min_frac
        self._chunked_window_local_origin = bool(window_local_origin)
        self._chunked_txt_groups = int(txt_groups)
        self._chunked_txt_len = int(txt_len)

    def _restore_standard_processors(self):
        """Swap all attention blocks back to FluxAttnProcessor."""
        for block in self.transformer_blocks:
            block.attn.set_processor(FluxAttnProcessor())
        for block in self.single_transformer_blocks:
            block.attn.set_processor(FluxAttnProcessor())
        self._chunked_proc_active = False
        self._chunked_window = -1
        self._chunked_text_min_frac = None
        self._chunked_window_local_origin = False
        self._chunked_txt_groups = 1
        self._chunked_txt_len = -1

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        target_resolution: Optional[int] = None,
        # ── HSWA: per-step parameters injected by run_flux.py's hswa_step_callback
        # The transformer keeps its sega/NTK auto-computation as the default
        # behaviour; when ntk_factor is set the HSWA scheduler is in control.
        ntk_factor: Optional[float] = None,
        window_size: Optional[int] = None,
        overlap_padding: int = 0,
        optimized: bool = False,
        attn_scale_floor: float = 1.0,
        max_patches: int = 0,
        window_floor: int = 0,
        attn_scale_power: float = 0.0,
        local_text_pos: bool = False,
        debug_windows: bool = False,
        window_text_min_frac: Optional[float] = None,
        window_local_origin: bool = False,
        # RoPE method: "ntk" -> NTK only (bypass sega even if ntk_factor > 1),
        # "sega" -> full sega-new behaviour, "none" -> force ntk_factor = 1,
        # "yarn" -> YaRN on spatial axes (see FluxPosEmbed, _use_yarn),
        # "dype" -> timestep-sharpened NTK factor (no SEGA spectral mscale).
        rope_method: str = "sega",
        spread_min: Optional[float] = None,
        spread_max: Optional[float] = None,
        mscale_floor: Optional[float] = None,
        mscale_ceiling: Optional[float] = None,
        force_global_sega_fft: bool = False,
        # Per-window conditioning: encoder_hidden_states holds ``window_text_groups``
        # equal-length token groups laid out along the sequence dim; group i is the
        # conditioning (text and/or image tokens) for attention window i.
        window_text_groups: int = 1,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        rope_mode: Optional[str] = None
        rope_f: Optional[float] = None
        s_sega_override: Optional[float] = None
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
            rope_mode = joint_attention_kwargs.pop("rope_simulate_mode", None)
            rope_f = joint_attention_kwargs.pop("rope_simulate_factor", None)
            legacy = joint_attention_kwargs.pop("sega_base_mscale_res_ratio", None)
            if legacy is not None:
                rope_f = float(legacy) if rope_f is None else rope_f
                rope_mode = rope_mode or "both"
        else:
            lora_scale = 1.0

        if rope_mode is not None:
            rope_mode = str(rope_mode).strip().lower()

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        hidden_states = self.x_embedder(hidden_states)

        # Flow-Match timestep (before *1000) for DyPE-style NTK modulation (`base/dype-flux`, ablation).
        ts_flow = float(timestep.flatten()[0].item()) if timestep is not None else 1.0
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections) if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        # Dynamic NTK factor
        max_h = int(img_ids[:, 1].max().item() - img_ids[:, 1].min().item()) + 1
        max_w = int(img_ids[:, 2].max().item() - img_ids[:, 2].min().item()) + 1
        if target_resolution is None:
            target_resolution = max(max_h, max_w) * 16
        d = self._rope_spatial_dim
        natural_s = max(float(target_resolution) / float(self.training_resolution), 1.0)
        F = max(float(rope_f), 1.0) if rope_f is not None else None

        if rope_mode not in (None, "", "natural", "none"):
            if rope_mode == "sega_only":
                s_ntk = natural_s
                s_sega_override = F
            elif rope_mode == "ntk_only":
                s_ntk = F if F is not None else natural_s
            elif rope_mode == "both":
                s_ntk = F if F is not None else natural_s
                s_sega_override = F
            else:
                s_ntk = natural_s
            s_ntk = max(float(s_ntk), 1.0)
            current_ntk_factor = s_ntk ** (2 * d / (d - 2)) / (1 + 0.1 * math.log(max(s_ntk, 1.0)))
        else:
            s = natural_s
            current_ntk_factor = s ** (2 * d / (d - 2)) / (1 + 0.1 * math.log(max(s, 1.0)))

        dynamic_spread = None
        energy_profile_h = None
        energy_profile_w = None

        # ================================================================
        # Visualization run modes (set by run_flux.py)
        #   baseline:  disable NTK + disable SEGA
        #   ntk_only:  enable NTK + disable SEGA
        #   ntk_sega:  current behaviour
        # ================================================================
        viz_mode = getattr(self, "visualize_mode", None)
        if viz_mode in ("baseline", "baseline_train1024", "baseline_gt", "yarn"):
            current_ntk_factor = 1.0

        # ── HSWA: per-step rope_method + ntk_factor override ─────────────────
        # When the HSWA scheduler is in charge it passes ntk_factor explicitly;
        # we honour it.  rope_method routes between sega-new (default), pure
        # NTK, and "none" (no scaling).
        assert rope_method in ROPE_METHODS, f"rope_method must be one of {ROPE_METHODS}"
        if rope_method == "none":
            current_ntk_factor = 1.0
        if ntk_factor is not None:
            current_ntk_factor = float(ntk_factor)

        # DyPE (`sega/flux_static_scale_ablation`, `base/dype-flux`): modulate effective NTK by timestep.
        if rope_method == "dype" and current_ntk_factor > 1.0:
            current_ntk_factor = max(
                1.0, float(current_ntk_factor) ** (2.0 * (ts_flow ** 2))
            )

        # Ease scalar extrapolation toward identity as denoising finishes: same
        # ``1 + r·(x-1)`` linear blend as SEGA excess (``compute_sega_excess_blend``),
        # but ``r`` is ``_viz_diffusion_time`` from the pipeline (1 = first step, 0 = last).
        r_den = float(getattr(self, "_viz_diffusion_time", 1.0))
        r_den = max(0.0, min(1.0, r_den))
        if rope_method in ("ntk", "yarn", "dype") and current_ntk_factor > 1.0:
            current_ntk_factor = 1.0 + r_den * (current_ntk_factor - 1.0)

        # Optional per-step overrides for the sega spread / mscale clamps so
        # the run_flux.py callback can sweep them without re-instantiating the
        # transformer.
        _spread_min = self.spread_min if spread_min is None else float(spread_min)
        _spread_max = self.spread_max if spread_max is None else float(spread_max)

        # When rope_method != "sega", bypass sega entirely (NTK-only path).
        _sega_bypass = rope_method != "sega"

        per_window_sega = (
            bool(optimized)
            and window_size is not None
            and not _sega_bypass
            and float(current_ntk_factor) > 1.0
            and not bool(local_text_pos)
            and not bool(window_local_origin)
            and not bool(force_global_sega_fft)
        )

        if viz_mode in ("baseline", "baseline_train1024", "baseline_gt", "yarn", "ntk_only"):
            dynamic_spread = None
            energy_profile_h = energy_profile_w = None
        elif _sega_bypass:
            dynamic_spread = None
            energy_profile_h = energy_profile_w = None
        elif (current_ntk_factor > 1.0 or s_sega_override is not None) and not per_window_sega:
            n_spatial = hidden_states.shape[1]
            if max_h * max_w == n_spatial:
                img_h, img_w = max_h, max_w
            else:
                aspect = max_h / max_w
                img_w = int(math.sqrt(n_spatial / aspect))
                img_h = n_spatial // max(1, img_w)

            # Per-axis spectral profiles
            n_bins_h = max(img_h // 2, 8)
            n_bins_w = max(img_w // 2, 8)
            energy_profile_h, energy_profile_w = compute_axis_spectral_profiles(
                hidden_states, img_h, img_w, n_bins_h, n_bins_w,
            )
            # Isotropic profile for spread schedule
            iso_profile = compute_spectral_energy_profile(
                hidden_states, img_h, img_w, n_bins=max(img_h, img_w) // 2,
            )
            # Visualization-only: record isotropic FFT energy mapped to RoPE bands.
            try:
                dim = int(self.pos_embed.axes_dim[2])
                dim_indices = torch.arange(0, dim, 2, dtype=torch.float64, device=hidden_states.device)
                scaled_theta = self.pos_embed.theta * current_ntk_factor
                iso_freqs = 1.0 / (scaled_theta ** (dim_indices / dim))
                _record_isotropic_fft_heatmap(
                    viz_collector=getattr(self, "viz_collector", None),
                    viz_step=int(getattr(self, "_viz_step", -1)),
                    freqs=iso_freqs,
                    iso_profile=iso_profile,
                )
            except Exception:
                pass
            dynamic_spread = compute_dynamic_spread(
                iso_profile, spread_min=_spread_min,
                spread_max=_spread_max, alpha=self.spread_alpha,
            )

        # Record continuous-frequency FFT energy (flux-ntk/flux-yarn style), independent of SEGA.
        coll = getattr(self, "viz_collector", None)
        step_idx = int(getattr(self, "_viz_step", -1))
        if coll is not None and step_idx >= 0 and hasattr(coll, "record_fft") and not per_window_sega:
            try:
                n_spatial = hidden_states.shape[1]
                if max_h * max_w == n_spatial:
                    img_h, img_w = max_h, max_w
                else:
                    aspect = max_h / max(1, max_w)
                    img_w = max(1, int(math.sqrt(n_spatial / max(aspect, 1e-8))))
                    img_h = max(1, n_spatial // img_w)

                n_freq = 128
                axis = compute_axis_spectral_profiles_continuous(hidden_states, height=img_h, width=img_w, n_freq=n_freq)
                iso_c = compute_isotropic_spectral_profile_continuous(hidden_states, height=img_h, width=img_w, n_freq=n_freq)
                diffusion_time = float(getattr(self, "_viz_diffusion_time", 1.0))

                def _rec(axis_name: str, freq: torch.Tensor, energy: torch.Tensor) -> None:
                    coll.record_fft(
                        step=int(step_idx),
                        diffusion_time=float(diffusion_time),
                        axis=str(axis_name),
                        freq=freq.detach().cpu().float().tolist(),
                        energy=energy.detach().cpu().float().tolist(),
                        min=float(energy.min().item()),
                        mean=float(energy.mean().item()),
                        max=float(energy.max().item()),
                    )

                _rec("height", axis["height"]["freq"], axis["height"]["energy"])
                _rec("width", axis["width"]["freq"], axis["width"]["energy"])
                _rec("isotropic", iso_c["freq"], iso_c["energy"])
            except Exception:
                pass

        ids = torch.cat((txt_ids, img_ids), dim=0)

        # Propagate visualization context to RoPE module (for per-step logging).
        # HSWA: also force-disable mscale when rope_method != "sega" so the
        # NTK-only path produces no base_mscale either.
        _disable_mscale = bool(
            viz_mode in ("baseline", "baseline_train1024", "baseline_gt", "yarn", "ntk_only")
            or _sega_bypass
        )
        try:
            if per_window_sega:
                self.pos_embed._viz_collector = None
                self.pos_embed._viz_step = -1
            else:
                self.pos_embed._viz_collector = getattr(self, "viz_collector", None)
                self.pos_embed._viz_step = int(getattr(self, "_viz_step", -1))
            self.pos_embed._disable_mscale = _disable_mscale
            self.pos_embed._use_yarn = bool(
                viz_mode == "yarn" or rope_method == "yarn"
            )
        except Exception:
            pass

        sega_excess_blend = compute_sega_excess_blend(
            window_size, window_floor, max_patches, rope_method=rope_method,
        )

        txt_len_val = int(txt_ids.shape[0])
        if int(window_text_groups) > 1 and (window_size is None or not optimized):
            raise ValueError(
                "window_text_groups > 1 requires optimized chunked attention with an active window",
            )
        if per_window_sega:
            image_rotary_emb = None
        elif (
            bool(window_local_origin)
            and window_size is not None
            and bool(optimized)
            and rope_method != "sega"
            and _uniform_window_tiles(
                img_ids, int(window_size), overlap_padding=int(overlap_padding),
            )
        ):
            # One window-local RoPE table per denoise step (shared by all attention layers).
            win_idx = _build_window_index(
                img_ids, int(window_size), overlap_padding=0,
            )[0].to(device=hidden_states.device)
            image_rotary_emb = compute_window_local_rope_emb(
                pos_embed=self.pos_embed,
                img_ids=img_ids,
                win_idx=win_idx,
                n_text=txt_len_val,
                window_size=int(window_size),
                rope_method=str(rope_method),
                spread_min=float(_spread_min),
                spread_max=float(_spread_max),
                spread_alpha=float(self.spread_alpha),
                window_floor=int(window_floor),
                max_patches=int(max_patches),
            )
        elif bool(window_local_origin):
            # Non-uniform tiles or per-window SEGA: processors build RoPE per window.
            image_rotary_emb = None
        else:
            image_rotary_emb = self.pos_embed(
                ids,
                ntk_factor=current_ntk_factor,
                mscale_spread=dynamic_spread,
                energy_profile_h=energy_profile_h,
                energy_profile_w=energy_profile_w,
                target_res=target_resolution,
                sega_excess_blend=sega_excess_blend,
                sega_base_mscale_res_ratio=s_sega_override,
            )

        # Propagate visualization context to attention processor.
        if hasattr(self, "viz_collector"):
            try:
                FluxAttnProcessor._viz_collector = self.viz_collector
                FluxAttnProcessor._viz_step = int(getattr(self, "_viz_step", -1))
                # Provide text/image layout so the processor can build spatial attention maps.
                FluxAttnProcessor._viz_txt_len = int(encoder_hidden_states.shape[1])
                FluxAttnProcessor._viz_img_hw = (int(max_h), int(max_w))
            except Exception:
                pass

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # ── HSWA: install chunked processors or build dense window mask ──────
        # Both paths are skipped when window_size is None — that is the plain
        # sega/flux_new path, preserved bit-identically when --hswa off.
        attn_mask = None
        if window_size is not None:
            if optimized:
                _text_frac = window_text_min_frac
                _local_origin = bool(window_local_origin)
                _dwt_on = bool(getattr(self, "_hswa_dwt_windows", False))
                _dwt_viz = bool(getattr(self, "_dwt_plot_prefix", None))
                dwt_leaves = None
                if _dwt_on or _dwt_viz:
                    leaves_computed, _ = self._compute_hswa_dwt_leaves(
                        hidden_states,
                        img_h=max_h,
                        img_w=max_w,
                        window_size=int(window_size),
                        window_floor=int(window_floor),
                        max_patches=int(max_patches),
                        txt_len=int(txt_len_val),
                        rope_method=str(rope_method),
                        img_ids=img_ids,
                    )
                    if _dwt_on:
                        dwt_leaves = leaves_computed if leaves_computed else None
                _txt_groups = max(1, int(window_text_groups))
                if (not self._chunked_proc_active
                        or self._chunked_window != window_size
                        or self._chunked_text_min_frac != _text_frac
                        or self._chunked_window_local_origin != _local_origin
                        or getattr(self, "_chunked_txt_groups", 1) != _txt_groups
                        or getattr(self, "_chunked_txt_len", -1) != txt_len_val
                        or _dwt_on):
                    self._install_chunked_processors(
                        txt_len=txt_len_val,
                        img_ids=img_ids,
                        window_size=window_size,
                        overlap_padding=overlap_padding,
                        attn_scale_floor=attn_scale_floor,
                        local_text_pos=local_text_pos,
                        ntk_factor=current_ntk_factor,
                        rope_method=rope_method,
                        debug_windows=debug_windows,
                        per_window_sega=per_window_sega,
                        txt_ids=txt_ids,
                        target_resolution=int(target_resolution),
                        window_floor=int(window_floor),
                        max_patches=int(max_patches),
                        spread_min=float(_spread_min),
                        spread_max=float(_spread_max),
                        spread_alpha=float(self.spread_alpha),
                        window_text_min_frac=_text_frac,
                        window_local_origin=_local_origin,
                        dwt_window_leaves=dwt_leaves,
                        txt_groups=_txt_groups,
                    )
                # No mask needed — chunked processor handles the restriction.
            else:
                if self._chunked_proc_active:
                    self._restore_standard_processors()

                cache_key = (window_size, overlap_padding, txt_len_val, img_ids.shape[0])
                if not hasattr(self, "_mask_cache"):
                    self._mask_cache: dict = {}
                    self._mask_cache_window: int = -1
                if window_size != self._mask_cache_window:
                    self._mask_cache.clear()
                    self._mask_cache_window = window_size
                if cache_key not in self._mask_cache:
                    self._mask_cache[cache_key] = build_sliding_window_mask(
                        img_ids=img_ids,
                        txt_len=txt_len_val,
                        window_size=window_size,
                        overlap_padding=overlap_padding,
                        device=hidden_states.device,
                    )
                attn_mask = self._mask_cache[cache_key]
        else:
            if self._chunked_proc_active:
                self._restore_standard_processors()

        # When using a dense window mask the FluxTransformerBlock.forward
        # expects it via joint_attention_kwargs (which the FluxAttention
        # forward filters and forwards to the processor's __call__).
        if attn_mask is not None:
            joint_attention_kwargs = (joint_attention_kwargs or {}).copy()
            joint_attention_kwargs["attention_mask"] = attn_mask

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                    temb=temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs,
                )
            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                    temb=temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs,
                )
            if controlnet_single_block_samples is not None:
                interval_control = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
