"""Compare DDIMSchedulerAdapter to diffusers.DDIMScheduler (set_timesteps + step)."""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler, DDPMScheduler

from find_delft_assignment.sampling.scheduler import DDIMSchedulerAdapter


def _schedulers(num_train: int = 1000, num_inference: int = 45):
    ddpm = DDPMScheduler(
        num_train_timesteps=num_train,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
    )
    ref = DDIMScheduler.from_config(ddpm.config)
    adapter = DDIMSchedulerAdapter(ddpm)
    return ref, adapter


def test_set_timesteps_matches_diffusers_cpu():
    ref, ours = _schedulers()
    ref.set_timesteps(45, device="cpu")
    ours.set_timesteps(45, device="cpu")
    assert ours.timesteps.shape == ref.timesteps.shape
    assert ours.timesteps.dtype == ref.timesteps.dtype
    torch.testing.assert_close(ours.timesteps, ref.timesteps)


def test_set_timesteps_matches_diffusers_cuda_if_available():
    if not torch.cuda.is_available():
        return
    ref, ours = _schedulers()
    ref.set_timesteps(45, device="cuda")
    ours.set_timesteps(45, device="cuda")
    torch.testing.assert_close(ours.timesteps, ref.timesteps)


def test_step_matches_diffusers_eta_zero():
    ref, ours = _schedulers()
    ref.set_timesteps(45, device="cpu")
    ours.set_timesteps(45, device="cpu")

    torch.manual_seed(0)
    sample = torch.randn(2, 3, 8, 8, dtype=torch.float32)
    noise_pred = torch.randn(2, 3, 8, 8, dtype=torch.float32)

    for t in (968, 500, 22, 0):
        t_int = int(t)
        out_ref = ref.step(noise_pred, t_int, sample, eta=0.0, generator=None)
        out_ours = ours.step(noise_pred, t_int, sample, eta=0.0, generator=None)
        torch.testing.assert_close(out_ours.prev_sample, out_ref.prev_sample, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            out_ours.pred_original_sample, out_ref.pred_original_sample, rtol=1e-5, atol=1e-6
        )


def test_step_matches_diffusers_eta_positive_same_generator():
    ref, ours = _schedulers()
    ref.set_timesteps(45, device="cpu")
    ours.set_timesteps(45, device="cpu")

    torch.manual_seed(1)
    sample = torch.randn(1, 3, 4, 4, dtype=torch.float32)
    noise_pred = torch.randn(1, 3, 4, 4, dtype=torch.float32)
    t_int = 968

    g_ref = torch.Generator(device="cpu").manual_seed(42)
    g_ours = torch.Generator(device="cpu").manual_seed(42)

    out_ref = ref.step(noise_pred, t_int, sample, eta=0.5, generator=g_ref)
    out_ours = ours.step(noise_pred, t_int, sample, eta=0.5, generator=g_ours)
    torch.testing.assert_close(out_ours.prev_sample, out_ref.prev_sample, rtol=1e-5, atol=1e-6)
