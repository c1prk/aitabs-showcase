"""Tests for the trainable PCEN front-end and its attachment to Kong."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aitabs.pipeline.audio.pcen import TrainablePCEN, attach_pcen_frontend


def test_pcen_shape_and_finite():
    pcen = TrainablePCEN(num_bands=16)
    E = torch.rand(2, 1, 50, 16).abs()  # non-negative mel energy
    out = pcen(E)
    assert out.shape == E.shape
    assert torch.isfinite(out).all()


def test_pcen_params_receive_gradient():
    pcen = TrainablePCEN(num_bands=8)
    E = torch.rand(1, 1, 20, 8).abs() + 0.1
    out = pcen(E)
    out.sum().backward()
    assert pcen._log_alpha.grad is not None
    assert pcen._log_delta.grad is not None
    assert pcen._log_r.grad is not None
    assert torch.isfinite(pcen._log_alpha.grad).all()


def test_pcen_trainable_flag_freezes():
    pcen = TrainablePCEN(num_bands=4, trainable=False)
    assert not pcen._log_alpha.requires_grad
    assert not pcen._log_r.requires_grad


def test_pcen_compresses_dynamic_range():
    # PCEN should reduce the ratio between loud and quiet frames (AGC behavior).
    pcen = TrainablePCEN(num_bands=4)
    loud = torch.full((1, 1, 10, 4), 10.0)
    quiet = torch.full((1, 1, 10, 4), 0.5)
    E = torch.cat([quiet, loud], dim=2)
    out = pcen(E)
    in_ratio = float(loud.mean() / quiet.mean())
    out_ratio = float(out[:, :, 10:, :].mean() / out[:, :, :10, :].mean().clamp(min=1e-6))
    assert out_ratio < in_ratio  # dynamic range compressed


def test_attach_to_kong_note_model_runs_and_backprops():
    pit = pytest.importorskip("piano_transcription_inference")
    from piano_transcription_inference.models import Note_pedal

    model = Note_pedal(frames_per_second=100, classes_num=88)
    pcen = attach_pcen_frontend(model.note_model)
    x = torch.randn(1, 16000)  # 1 s
    out = model.note_model(x)
    assert "frame_output" in out
    assert out["frame_output"].shape[-1] == 88
    out["frame_output"].sum().backward()
    assert pcen._log_alpha.grad is not None  # gradient reached the PCEN front-end
