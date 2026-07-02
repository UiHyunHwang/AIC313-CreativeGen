"""FLUX.2 transformer with mosaic tile attention.

Subclasses diffusers' ``Flux2Transformer2DModel``. Without tile kwargs the
forward is byte-identical to stock (delegates to ``super().forward``). In tile
mode the sequence is reorganised so every attention window is an independent
mini-generation:

  window i attends over  [ text_i (T) | window_i (K gen tokens) | ref_i (R) ]

- ``text_i``: the conditioning text stream, replicated per window after the
  context embedder — each tile is fully separated from the others in terms of
  text tokens (streams diverge through the blocks).
- ``ref_i``: optional packed VAE latents of one gallery image per window
  (FLUX.2-native reference conditioning, T-coordinate 10).
- RoPE is window-local: gen tokens get coordinates relative to their window
  origin, so each tile looks like a standalone image at native training scale.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    Flux2AttnProcessor,
    Flux2ParallelSelfAttnProcessor,
    Flux2Transformer2DModel,
    _get_qkv_projections,
)

REF_T_COORD = 10  # time coordinate of the first reference image (pipeline convention)


def _starts_axis(total: int, w: int) -> list:
    """Edge-flush balanced window start positions (same scheme as flux1)."""
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


def build_window_indices(ph: int, pw: int, ws: int) -> Tuple[torch.Tensor, int, int, int, int]:
    """Flat row-major token indices per window over a ``ph``×``pw`` patch grid.

    Every window is ``kh``×``kw`` (= ws except when the grid is smaller); edge
    windows are shifted flush so all windows are full-sized and may overlap.
    Returns (win_idx [n_windows, kh*kw], n_rows, n_cols, kh, kw).
    """
    rs = _starts_axis(ph, ws)
    cs = _starts_axis(pw, ws)
    kh, kw = min(ws, ph), min(ws, pw)
    wins = []
    for r0 in rs:
        for c0 in cs:
            rows = torch.arange(r0, r0 + kh)
            cols = torch.arange(c0, c0 + kw)
            wins.append((rows[:, None] * pw + cols[None, :]).reshape(-1))
    return torch.stack(wins), len(rs), len(cs), kh, kw


class Flux2TileContext:
    """Per-tile-pass geometry: window indices, token counts, local RoPE table."""

    def __init__(
        self,
        *,
        ph: int,
        pw: int,
        window_patches: int,
        txt_len: int,
        ref_grid: int,
        pos_embed,
        device: torch.device,
    ):
        self.key = (ph, pw, window_patches, txt_len, ref_grid)
        win_idx, n_rows, n_cols, kh, kw = build_window_indices(ph, pw, window_patches)
        self.win_idx = win_idx  # CPU [W, K]
        self.n_windows = win_idx.shape[0]
        self.n_rows, self.n_cols = n_rows, n_cols
        self.K = kh * kw
        self.T = txt_len
        self.R = ref_grid * ref_grid
        self.N = ph * pw

        # Window-local 4D ids (T, H, W, L): text on the L axis at origin, gen
        # tokens relative to the window origin, refs as reference image 0.
        txt_ids = torch.zeros(txt_len, 4)
        txt_ids[:, 3] = torch.arange(txt_len)
        gh = torch.arange(kh).repeat_interleave(kw)
        gw = torch.arange(kw).repeat(kh)
        gen_ids = torch.stack([torch.zeros(self.K), gh, gw, torch.zeros(self.K)], dim=1)
        parts = [txt_ids, gen_ids]
        if self.R > 0:
            rh = torch.arange(ref_grid).repeat_interleave(ref_grid)
            rw = torch.arange(ref_grid).repeat(ref_grid)
            ref_ids = torch.stack(
                [torch.full((self.R,), float(REF_T_COORD)), rh, rw, torch.zeros(self.R)], dim=1,
            )
            parts.append(ref_ids)
        ids = torch.cat(parts, dim=0).to(device)
        cos, sin = pos_embed(ids)
        self.rope = (cos, sin)

        # Scatter-mean weights for gen tokens (edge windows may overlap).
        count = torch.zeros(self.N)
        count.scatter_add_(0, win_idx.reshape(-1), torch.ones(win_idx.numel()))
        self.gen_count = count  # CPU [N]


class Flux2AttnProcessorChunked:
    """Dual-stream tile attention: encoder holds W per-window text streams."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, ctx: Flux2TileContext):
        self.ctx = ctx

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb=None,  # ignored — window-local RoPE from ctx
    ) -> torch.Tensor:
        ctx = self.ctx
        W, K, T, R, N = ctx.n_windows, ctx.K, ctx.T, ctx.R, ctx.N

        q, k, v, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
        v = v.unflatten(-1, (attn.heads, -1))
        eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
        ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
        ev = ev.unflatten(-1, (attn.heads, -1))

        B, _, H, D = q.shape
        dev = q.device
        flat = ctx.win_idx.reshape(-1).to(dev)

        def assemble(t_txt, t_img):
            gen, ref = t_img[:, :N], t_img[:, N:]
            parts = [
                t_txt.reshape(B, W, T, H, D),
                gen[:, flat].view(B, W, K, H, D),
            ]
            if R > 0:
                parts.append(ref.reshape(B, W, R, H, D))
            return torch.cat(parts, dim=2).view(B * W, T + K + R, H, D)

        q_w = assemble(eq, q)
        k_w = assemble(ek, k)
        v_w = assemble(ev, v)

        cos, sin = ctx.rope
        q_w = apply_rotary_emb(q_w, (cos, sin), sequence_dim=1)
        k_w = apply_rotary_emb(k_w, (cos, sin), sequence_dim=1)

        out = dispatch_attention_fn(
            q_w, k_w, v_w, attn_mask=None, backend=self._attention_backend,
        ).view(B, W, T + K + R, H, D)

        out_txt = out[:, :, :T].reshape(B, W * T, H, D)
        out_gen = torch.zeros(B, N, H, D, dtype=out.dtype, device=dev)
        idx = flat.view(1, -1, 1, 1).expand(B, -1, H, D)
        out_gen.scatter_add_(1, idx, out[:, :, T:T + K].reshape(B, W * K, H, D))
        out_gen = out_gen / ctx.gen_count.to(device=dev, dtype=out.dtype).clamp(min=1.0).view(1, N, 1, 1)
        if R > 0:
            out_ref = out[:, :, T + K:].reshape(B, W * R, H, D)
            img_out = torch.cat([out_gen, out_ref], dim=1)
        else:
            img_out = out_gen

        img_out = img_out.flatten(2, 3).to(q.dtype)
        txt_out = out_txt.flatten(2, 3).to(q.dtype)

        enc_out = attn.to_add_out(txt_out)
        img_out = attn.to_out[1](attn.to_out[0](img_out))
        return img_out, enc_out


class Flux2ParallelSelfAttnProcessorChunked:
    """Single-stream tile attention over the [W·T text | N gen | W·R ref] layout."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, ctx: Flux2TileContext):
        self.ctx = ctx

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb=None,  # ignored — window-local RoPE from ctx
    ) -> torch.Tensor:
        ctx = self.ctx
        W, K, T, R, N = ctx.n_windows, ctx.K, ctx.T, ctx.R, ctx.N

        hs = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden = torch.split(
            hs, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1,
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        B, S, H, D = query.shape
        dev = query.device
        flat = ctx.win_idx.reshape(-1).to(dev)

        def assemble(x):
            txt, gen, ref = x[:, :W * T], x[:, W * T:W * T + N], x[:, W * T + N:]
            parts = [
                txt.reshape(B, W, T, H, D),
                gen[:, flat].view(B, W, K, H, D),
            ]
            if R > 0:
                parts.append(ref.reshape(B, W, R, H, D))
            return torch.cat(parts, dim=2).view(B * W, T + K + R, H, D)

        q_w = assemble(query)
        k_w = assemble(key)
        v_w = assemble(value)

        cos, sin = ctx.rope
        q_w = apply_rotary_emb(q_w, (cos, sin), sequence_dim=1)
        k_w = apply_rotary_emb(k_w, (cos, sin), sequence_dim=1)

        out = dispatch_attention_fn(
            q_w, k_w, v_w, attn_mask=None, backend=self._attention_backend,
        ).view(B, W, T + K + R, H, D)

        out_full = torch.zeros(B, S, H, D, dtype=out.dtype, device=dev)
        out_full[:, :W * T] = out[:, :, :T].reshape(B, W * T, H, D)
        idx = (flat + W * T).view(1, -1, 1, 1).expand(B, -1, H, D)
        out_full.scatter_add_(1, idx, out[:, :, T:T + K].reshape(B, W * K, H, D))
        cnt = ctx.gen_count.to(device=dev, dtype=out.dtype).clamp(min=1.0)
        out_full[:, W * T:W * T + N] = out_full[:, W * T:W * T + N] / cnt.view(1, N, 1, 1)
        if R > 0:
            out_full[:, W * T + N:] = out[:, :, T + K:].reshape(B, W * R, H, D)

        out_full = out_full.flatten(2, 3).to(query.dtype)
        mlp_hidden = attn.mlp_act_fn(mlp_hidden)
        return attn.to_out(torch.cat([out_full, mlp_hidden], dim=-1))


class Flux2Transformer2DModelMosaic(Flux2Transformer2DModel):
    """Stock FLUX.2 transformer + tile mode (per-window text/ref conditioning)."""

    _tile_ctx: Optional[Flux2TileContext] = None

    def _install_tile_processors(self, ctx: Flux2TileContext) -> None:
        for block in self.transformer_blocks:
            block.attn.set_processor(Flux2AttnProcessorChunked(ctx))
        for block in self.single_transformer_blocks:
            block.attn.set_processor(Flux2ParallelSelfAttnProcessorChunked(ctx))
        self._tile_ctx = ctx

    def _restore_stock_processors(self) -> None:
        for block in self.transformer_blocks:
            block.attn.set_processor(Flux2AttnProcessor())
        for block in self.single_transformer_blocks:
            block.attn.set_processor(Flux2ParallelSelfAttnProcessor())
        self._tile_ctx = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[dict] = None,
        return_dict: bool = True,
        # ── tile mode (mosaic) ────────────────────────────────────────────
        tile_window_patches: Optional[int] = None,
        tile_grid: Optional[Tuple[int, int]] = None,
        tile_ref_states: Optional[torch.Tensor] = None,  # [W, R, in_channels]
        tile_ref_grid: int = 0,
    ):
        if tile_window_patches is None:
            if self._tile_ctx is not None:
                self._restore_stock_processors()
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
            )

        if tile_grid is None:
            raise ValueError("tile mode requires tile_grid=(patch_h, patch_w)")
        ph, pw = tile_grid
        B, N, _ = hidden_states.shape
        if N != ph * pw:
            raise ValueError(f"hidden_states has {N} gen tokens, tile_grid implies {ph * pw}")
        T = encoder_hidden_states.shape[1]
        ref_grid = int(tile_ref_grid) if tile_ref_states is not None else 0

        key = (ph, pw, int(tile_window_patches), T, ref_grid)
        ctx = self._tile_ctx
        if ctx is None or ctx.key != key:
            ctx = Flux2TileContext(
                ph=ph, pw=pw, window_patches=int(tile_window_patches),
                txt_len=T, ref_grid=ref_grid,
                pos_embed=self.pos_embed, device=hidden_states.device,
            )
            self._install_tile_processors(ctx)
        W = ctx.n_windows

        if tile_ref_states is not None:
            if tile_ref_states.shape[0] != W:
                raise ValueError(
                    f"tile_ref_states has {tile_ref_states.shape[0]} windows, expected {W}",
                )
            refs = tile_ref_states.reshape(1, -1, tile_ref_states.shape[-1])
            refs = refs.to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = torch.cat([hidden_states, refs.expand(B, -1, -1)], dim=1)

        # ── stock embedding path (timestep, modulation, projections) ─────
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        # Per-window text separation: W independent copies of the text stream.
        encoder_hidden_states = encoder_hidden_states.repeat(1, W, 1)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=None,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=None,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        hidden_states = hidden_states[:, W * T:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
