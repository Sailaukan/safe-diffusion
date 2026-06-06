# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LossBreakdown:
    loss: torch.Tensor
    token_loss: torch.Tensor
    diffusion_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    loss_mask: torch.Tensor


class LogLinearNoiseSchedule(nn.Module):
    """Log-linear schedule used by continuous-time MDLM."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.total_noise(t), self.rate_noise(t)

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1 - self.eps) * t)


class MaskedDiscreteDiffusion(nn.Module):
    """SAFE MDLM engine using absorbing-state diffusion with SUBS parameterization."""

    def __init__(
        self,
        vocab_size: int,
        mask_index: int,
        pad_index: int,
        bos_index: int,
        eos_index: int,
        time_distribution,
        noise_schedule: LogLinearNoiseSchedule,
        freeze_special_tokens: bool = True,
        sampling_steps: int = 128,
        sampling_eps: float = 1e-3,
        final_denoise: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.pad_index = pad_index
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.time_distribution = time_distribution
        self.noise_schedule = noise_schedule
        self.freeze_special_tokens = freeze_special_tokens
        self.sampling_steps = sampling_steps
        self.sampling_eps = sampling_eps
        self.final_denoise = final_denoise
        self.neg_infinity = -1_000_000.0
        self.register_buffer("device_anchor", torch.zeros((), dtype=torch.float32), persistent=False)

    def to_device(self, device: torch.device | str) -> None:
        self.to(device)

    def sample_time(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        if device is None:
            device = self.device_anchor.device
        return self.time_distribution.sample(batch_size, device=device)

    def time_conditioning(self, t: torch.Tensor) -> torch.Tensor:
        sigma, _ = self.noise_schedule(t)
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        return sigma

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        sigma = self.time_conditioning(t)
        return torch.exp(-sigma)

    def get_frozen_token_mask(self, x: torch.Tensor) -> torch.Tensor:
        if not self.freeze_special_tokens:
            return torch.zeros_like(x, dtype=torch.bool)
        frozen_mask = x.eq(self.pad_index)
        frozen_mask |= x.eq(self.bos_index)
        frozen_mask |= x.eq(self.eos_index)
        return frozen_mask

    def _q_xt(self, x: torch.Tensor, move_chance: torch.Tensor) -> torch.Tensor:
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        return torch.where(move_indices, self.mask_index, x)

    def forward_process(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        frozen_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma = self.time_conditioning(t)
        move_chance = (1 - torch.exp(-sigma))[:, None]
        xt = self._q_xt(x0, move_chance)
        if frozen_mask is not None:
            xt = torch.where(frozen_mask, x0, xt)
        return xt

    def _subs_log_probs(self, logits: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        logits[..., self.mask_index] = self.neg_infinity

        copy_mask = xt.ne(self.mask_index)
        if copy_mask.any():
            logits[copy_mask] = self.neg_infinity
            logits[copy_mask, xt[copy_mask]] = 0.0

        return logits.log_softmax(dim=-1)

    def _token_losses(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma, dsigma = self.noise_schedule(t)
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if dsigma.ndim > 1:
            dsigma = dsigma.squeeze(-1)

        log_probs = self._subs_log_probs(logits, xt)
        log_p_theta = torch.gather(log_probs, -1, x0[..., None]).squeeze(-1)
        diffusion_loss = -log_p_theta * (dsigma / torch.expm1(sigma).clamp_min(1e-12))[:, None]
        reconstruction_loss = -torch.gather(log_probs, -1, x0[..., None]).squeeze(-1)
        token_loss = diffusion_loss
        return token_loss, diffusion_loss, reconstruction_loss

    def loss_terms(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        frozen_mask: torch.Tensor | None = None,
        global_mean: bool = False,
    ) -> LossBreakdown:
        token_loss, diffusion_loss, reconstruction_loss = self._token_losses(logits, x0, xt, t)

        if mask is None:
            loss_mask = torch.ones_like(x0, dtype=torch.bool)
        else:
            loss_mask = mask.to(torch.bool)
        if frozen_mask is not None:
            loss_mask &= ~frozen_mask

        masked_token_loss = token_loss * loss_mask
        if global_mean:
            loss = masked_token_loss.sum() / loss_mask.sum().clamp_min(1)
        else:
            loss = masked_token_loss.sum(dim=-1) / loss_mask.sum(dim=-1).clamp_min(1)

        return LossBreakdown(
            loss=loss,
            token_loss=token_loss,
            diffusion_loss=diffusion_loss,
            reconstruction_loss=reconstruction_loss,
            loss_mask=loss_mask,
        )

    def loss(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        global_mean: bool = False,
        frozen_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss_terms(
            logits=logits,
            x0=x0,
            xt=xt,
            t=t,
            mask=mask,
            frozen_mask=frozen_mask,
            global_mean=global_mean,
        ).loss

    def initialize_sample(self, template: torch.Tensor, editable_mask: torch.Tensor) -> torch.Tensor:
        xt = template.clone()
        xt[editable_mask] = self.mask_index
        return xt

    def get_num_steps_confidence(self, _: torch.Tensor | None = None) -> int:
        return max(int(self.sampling_steps), 2)

    def get_sampling_timesteps(self, device: torch.device | str, num_steps: int | None = None) -> torch.Tensor:
        num_steps = self.get_num_steps_confidence() if num_steps is None else max(int(num_steps), 2)
        return torch.linspace(1.0, self.sampling_eps, num_steps + 1, device=device)

    def _sample_with_randomness(self, probs: torch.Tensor, randomness: float) -> torch.Tensor:
        if randomness <= 0:
            return probs.argmax(dim=-1)
        log_probs = probs.clamp_min(1e-12).log()
        gumbel = -torch.log(-torch.log(torch.rand_like(log_probs).clamp_min(1e-12)))
        return (log_probs + randomness * gumbel).argmax(dim=-1)

    def degrade_context(self, x: torch.Tensor, frozen_mask: torch.Tensor, gamma: float) -> torch.Tensor:
        if gamma <= 0:
            return x

        editable_context = frozen_mask.clone()
        editable_context &= ~x.eq(self.pad_index)
        editable_context &= ~x.eq(self.bos_index)
        editable_context &= ~x.eq(self.eos_index)

        degraded = x.clone()
        for batch_idx in range(x.shape[0]):
            candidate_ids = editable_context[batch_idx].nonzero(as_tuple=True)[0]
            if candidate_ids.numel() == 0:
                continue
            num_replace = min(candidate_ids.numel(), int(candidate_ids.numel() * gamma))
            if num_replace == 0 and gamma > 0:
                num_replace = 1
            if num_replace == 0:
                continue
            selected = candidate_ids[torch.randperm(candidate_ids.numel(), device=x.device)[:num_replace]]
            degraded[batch_idx, selected] = self.mask_index
        return degraded

    def _move_chance(self, t: torch.Tensor) -> torch.Tensor:
        sigma = self.time_conditioning(t)
        return (1 - torch.exp(-sigma))[:, None, None]

    def _absorbing_posterior(
        self,
        logits: torch.Tensor,
        xt: torch.Tensor,
        move_chance_t: torch.Tensor,
        move_chance_s: torch.Tensor,
        softmax_temp: float = 1.0,
    ) -> torch.Tensor:
        log_x_theta = self._subs_log_probs(logits / max(softmax_temp, 1e-6), xt)
        x_theta = log_x_theta.exp()
        q_xs = x_theta * (move_chance_t - move_chance_s)
        q_xs[..., self.mask_index] = move_chance_s[..., 0]
        q_xs = q_xs / move_chance_t.clamp_min(1e-12)

        copy_mask = xt.ne(self.mask_index)
        if copy_mask.any():
            q_xs[copy_mask] = 0.0
            q_xs[copy_mask, xt[copy_mask]] = 1.0

        return q_xs.clamp_min(0.0)

    def step_confidence(
        self,
        logits: torch.Tensor,
        xt: torch.Tensor,
        step_idx: int,
        num_steps: int,
        softmax_temp: float = 1.0,
        randomness: float = 1.0,
        editable_mask: torch.Tensor | None = None,
        timestep_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timestep_grid is None:
            timestep_grid = self.get_sampling_timesteps(xt.device, num_steps=num_steps)

        t = timestep_grid[step_idx].expand(xt.shape[0])
        s = timestep_grid[step_idx + 1].expand(xt.shape[0])
        posterior = self._absorbing_posterior(
            logits=logits,
            xt=xt,
            move_chance_t=self._move_chance(t),
            move_chance_s=self._move_chance(s),
            softmax_temp=softmax_temp,
        )
        xs = self._sample_with_randomness(posterior, randomness=randomness)

        if editable_mask is None:
            return xs
        return torch.where(editable_mask, xs, xt)

    def final_denoise_step(
        self,
        logits: torch.Tensor,
        xt: torch.Tensor,
        editable_mask: torch.Tensor,
        softmax_temp: float = 1.0,
        randomness: float = 0.0,
    ) -> torch.Tensor:
        target_mask = editable_mask & xt.eq(self.mask_index)
        if not target_mask.any():
            return xt

        log_probs = self._subs_log_probs(logits / max(softmax_temp, 1e-6), xt)
        probs = log_probs.exp()
        xs = self._sample_with_randomness(probs, randomness=randomness)
        return torch.where(target_mask, xs, xt)
