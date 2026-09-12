"""Trainable PCEN front-end for the Kong note model.

PCEN (Per-Channel Energy Normalization; Wang, Lostanlen & Bello) replaces static
log-compression of the mel spectrogram with a *learnable* per-channel automatic
gain + dynamic-range compression. It is known to enhance onsets and add
loudness/reverb robustness — directly targeting the quiet/short recall bucket
that thresholds and the ensemble can't reach.

    PCEN(E) = ( E / (eps + M)^alpha + delta )^r - delta^r
    M[t]    = (1 - s) * M[t-1] + s * E[t]        (per-band causal smoother)

``alpha``, ``delta``, ``r`` are trainable per mel band (softplus-reparametrized to
stay positive); ``s`` (the smoother time constant) is a fixed hyperparameter, so
the smoother is a plain differentiable EMA — no differentiable-IIR-coefficient
headache, and the params that matter for compression are still learned.

Integration is deliberately minimal: Kong's ``note_model.forward`` computes
``x = self.logmel_extractor(power_spectrogram)`` and then runs the CRNN on ``x``.
:func:`attach_pcen_frontend` swaps ``logmel_extractor`` for :class:`MelPCEN`
(mel projection + PCEN), so PCEN replaces the log with zero changes to Kong's
forward. ⚠ The pretrained CRNN was trained on *log*-mel; PCEN output is a
different distribution, so ``bn0`` + the CRNN must adapt — warm up PCEN+bn0 first
or use a lower LR (see ``RESEARCH_PCEN.md``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) == y (for initializing softplus-reparam params)."""
    return math.log(math.expm1(y))


class TrainablePCEN(nn.Module):
    """Per-channel energy normalization with learnable alpha/delta/r.

    Operates on non-negative mel energy of shape ``(B, 1, T, M)`` and smooths
    along the time axis ``T``. ``s`` is fixed; ``alpha, delta, r`` are learned
    per band (size ``M``), initialized to standard PCEN values.
    """

    def __init__(
        self,
        num_bands: int,
        *,
        s: float = 0.025,
        alpha: float = 0.8,
        delta: float = 2.0,
        r: float = 0.5,
        eps: float = 1e-6,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.s = float(s)
        init = lambda v: torch.full((num_bands,), _inv_softplus(v), dtype=torch.float32)
        self._log_alpha = nn.Parameter(init(alpha), requires_grad=trainable)
        self._log_delta = nn.Parameter(init(delta), requires_grad=trainable)
        self._log_r = nn.Parameter(init(r), requires_grad=trainable)

    def _smooth(self, E: torch.Tensor) -> torch.Tensor:
        """Causal EMA along time (dim=2). Differentiable; loop over T frames."""
        s = self.s
        m = E[:, :, 0, :]
        outs = [m]
        for t in range(1, E.shape[2]):
            m = (1.0 - s) * m + s * E[:, :, t, :]
            outs.append(m)
        return torch.stack(outs, dim=2)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        E = torch.clamp(E, min=0.0)
        alpha = torch.nn.functional.softplus(self._log_alpha)
        delta = torch.nn.functional.softplus(self._log_delta)
        r = torch.nn.functional.softplus(self._log_r)
        M = self._smooth(E)
        smooth = (self.eps + M) ** alpha
        return (E / smooth + delta) ** r - delta ** r


class MelPCEN(nn.Module):
    """Mel projection (reusing Kong's frozen ``melW``) followed by trainable PCEN.

    Drop-in replacement for ``LogmelFilterBank``: takes the power spectrogram
    ``(B, 1, T, F)`` and returns a PCEN-compressed mel representation
    ``(B, 1, T, M)`` — so it slots straight into Kong's forward.
    """

    def __init__(self, orig_logmel: nn.Module, **pcen_kwargs) -> None:
        super().__init__()
        melW = orig_logmel.melW.detach().clone()
        self.register_buffer("melW", melW)
        self.pcen = TrainablePCEN(melW.shape[-1], **pcen_kwargs)

    def forward(self, power_spectrogram: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(power_spectrogram, self.melW)
        return self.pcen(mel)


def attach_pcen_frontend(note_model: nn.Module, **pcen_kwargs) -> TrainablePCEN:
    """Swap ``note_model.logmel_extractor`` for a trainable PCEN front-end.

    Returns the :class:`TrainablePCEN` so the caller can put its parameters in a
    dedicated optimizer group (typically a higher LR than the pretrained CRNN).
    """
    orig = note_model.logmel_extractor
    mel_pcen = MelPCEN(orig, **pcen_kwargs)
    note_model.logmel_extractor = mel_pcen
    return mel_pcen.pcen
