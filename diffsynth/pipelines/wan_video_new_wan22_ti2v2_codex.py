import types

import torch

from .wan_video_new import *  # noqa: F401,F403
from .wan_video_new import WanVideoPipeline as _WanVideoPipelineBase


class WanVideoPipeline(_WanVideoPipelineBase):
    """Wan2.2 TI2V2 prefix-training wrapper.

    This keeps the original WanVideoPipeline untouched and only replaces the
    training loss with WoVR-style 5-frame context + 8-frame horizon behavior.
    """

    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        input_latents = inputs["input_latents"]
        noise = inputs["noise"]
        if input_latents.shape[2] < 4:
            raise ValueError(
                "Wan2.2 TI2V2 prefix training expects at least 4 latent temporal slots. "
                f"Got input_latents shape {tuple(input_latents.shape)}."
            )

        latents = self.scheduler.add_noise(input_latents, noise, timestep)
        latents[:, :, 0:1] = input_latents[:, :, 0:1]

        low_noise_step_offset = int(inputs.get("ti2v2_low_noise_step_offset", 50))
        low_noise_idx = torch.tensor([max(0, len(self.scheduler.timesteps) - low_noise_step_offset)])
        low_noise_timestep = self.scheduler.timesteps[low_noise_idx].to(dtype=self.torch_dtype, device=self.device)
        latents[:, :, 1:2] = self.scheduler.add_noise(
            input_latents[:, :, 1:2],
            noise[:, :, 1:2],
            low_noise_timestep,
        )
        inputs["latents"] = latents

        training_target = self.scheduler.training_target(input_latents, noise, timestep)
        noise_pred = self.model_fn(**inputs, timestep=timestep)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="none")
        horizon_slots = int(inputs.get("ti2v2_horizon_loss_latent_slots", 2))
        if horizon_slots <= 0 or horizon_slots > loss.shape[2]:
            raise ValueError(f"Invalid horizon latent slots {horizon_slots} for loss shape {tuple(loss.shape)}.")
        mask = torch.zeros_like(loss)
        mask[:, :, -horizon_slots:] = 1.0
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    @staticmethod
    def from_pretrained(*args, **kwargs):
        pipe = _WanVideoPipelineBase.from_pretrained(*args, **kwargs)
        pipe.training_loss = types.MethodType(WanVideoPipeline.training_loss, pipe)
        return pipe
