import types

import torch

from .wan_video_new import *  # noqa: F401,F403
from .wan_video_new import (
    WanVideoPipeline as _WanVideoPipelineBase,
    WanVideoUnit_VACE_raymap as _WanVideoUnit_VACE_raymap,
)
from ..utils import PipelineUnit


class WanVideoUnit_Wan22CodexVACE144(PipelineUnit):
    """Build Wan2.2 Codex VACE controls from ray/action maps.

    Compatible modes:
    - vace_in_dim=144: VAE-encode ray_map_o, ray_map_d, and action_map/vace_video
      into 48 channels each.
    - vace_in_dim=54: VAE-encode only action_map/vace_video into 48 channels,
      then concatenate raw float ray_map_o and ray_map_d tensors with 3 channels each.
    - vace_in_dim=72: same as 54, but each ray tensor stacks four source frames
      per Wan VAE temporal slot, so ray_map_o and ray_map_d have 12 channels each.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "vace_video",
                "action_map",
                "ray_map_o",
                "ray_map_d",
                "vace_scale",
                "height",
                "width",
                "num_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def _as_latents(self, pipe, value, name, tiled, tile_size, tile_stride):
        if value is None:
            raise ValueError(f"{name} is required for Wan2.2 Codex VACE.")
        if torch.is_tensor(value):
            latents = value.to(dtype=pipe.torch_dtype, device=pipe.device)
            if latents.ndim == 4:
                latents = latents.unsqueeze(0)
            if latents.ndim != 5:
                raise ValueError(f"{name} tensor must have 4 or 5 dims, got {tuple(latents.shape)}.")
            raise NotImplementedError("vace_in_dim=144 expects raw image/video conditions and VAE-encodes them.")
        video = pipe.preprocess_video(value)
        return pipe.vae.encode(
            video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

    def _as_raw_ray_latents(self, pipe, value, name, expected_t_hw, expected_channels):
        if value is None:
            raise ValueError(f"{name} is required for Wan2.2 Codex VACE raw-ray mode.")
        if not torch.is_tensor(value):
            raise ValueError(f"{name} must be a raw float tensor for raw-ray mode, got {type(value)!r}.")
        latents = value.to(dtype=pipe.torch_dtype, device=pipe.device)
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        if latents.ndim != 5:
            raise ValueError(f"{name} tensor must have 4 or 5 dims, got {tuple(latents.shape)}.")
        if latents.shape[1] != expected_channels:
            raise ValueError(f"{name} must have {expected_channels} channels in raw-ray mode, got {latents.shape[1]}.")
        if tuple(latents.shape[2:]) != tuple(expected_t_hw):
            raise ValueError(f"{name} shape {tuple(latents.shape[2:])} must match action latent shape {tuple(expected_t_hw)}.")
        return latents

    def process(
        self,
        pipe,
        vace_video,
        action_map,
        ray_map_o,
        ray_map_d,
        vace_scale,
        height,
        width,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        del height, width, num_frames
        action_source = action_map if action_map is not None else vace_video
        if action_source is None and ray_map_o is None and ray_map_d is None:
            return {"vace_context": None, "vace_global_context": None, "vace_scale": vace_scale}
        if action_source is None or ray_map_o is None or ray_map_d is None:
            raise ValueError("Wan2.2 Codex VACE requires ray_map_o, ray_map_d, and action_map or vace_video.")

        pipe.load_models_to_device(["vae"])
        expected_in_dim = getattr(pipe.vace, "vace_in_dim", 144)
        action_latents = self._as_latents(pipe, action_source, "action_map/vace_video", tiled, tile_size, tile_stride)
        z_dim = getattr(pipe.vae, "z_dim", getattr(pipe.vae.model, "z_dim", 48))
        if action_latents.shape[1] != z_dim:
            raise ValueError(f"action_map/vace_video should have {z_dim} channels after VAE encode, got {action_latents.shape[1]}.")

        if expected_in_dim in (z_dim + 6, z_dim + 24):
            expected_ray_channels = 3 if expected_in_dim == z_dim + 6 else 12
            ray_o_latents = self._as_raw_ray_latents(pipe, ray_map_o, "ray_map_o", action_latents.shape[2:], expected_ray_channels)
            ray_d_latents = self._as_raw_ray_latents(pipe, ray_map_d, "ray_map_d", action_latents.shape[2:], expected_ray_channels)
            vace_context = torch.cat((action_latents, ray_o_latents, ray_d_latents), dim=1)
        elif expected_in_dim == z_dim * 3:
            ray_o_latents = self._as_latents(pipe, ray_map_o, "ray_map_o", tiled, tile_size, tile_stride)
            ray_d_latents = self._as_latents(pipe, ray_map_d, "ray_map_d", tiled, tile_size, tile_stride)
            shapes = {tuple(u.shape[2:]) for u in (ray_o_latents, ray_d_latents, action_latents)}
            if len(shapes) != 1:
                raise ValueError(
                    "ray/action latent temporal-spatial shapes must match, got "
                    f"{ray_o_latents.shape}, {ray_d_latents.shape}, {action_latents.shape}."
                )
            for name, latents in (("ray_map_o", ray_o_latents), ("ray_map_d", ray_d_latents)):
                if latents.shape[1] != z_dim:
                    raise ValueError(f"{name} should have {z_dim} channels after VAE encode, got {latents.shape[1]}.")
            vace_context = torch.cat((ray_o_latents, ray_d_latents, action_latents), dim=1)
        else:
            raise ValueError(f"Unsupported VACE input dim {expected_in_dim}; expected {z_dim + 6}, {z_dim + 24}, or {z_dim * 3}.")

        if vace_context.shape[1] != expected_in_dim:
            raise ValueError(f"Expected VACE input dim {expected_in_dim}, got {vace_context.shape[1]}.")
        return {"vace_context": vace_context, "vace_global_context": None, "vace_scale": vace_scale}


class WanVideoPipeline(_WanVideoPipelineBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._install_wan22_codex_units()

    def training_loss(self, **inputs):
        print(f'TI2V2_VACE_1context的training_loss被调用')
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        input_latents = inputs["input_latents"]
        noise = inputs["noise"]
        latents = self.scheduler.add_noise(input_latents, noise, timestep)
        if inputs.get("fuse_vae_embedding_in_latents", False):
            latents[:, :, 0:1] = input_latents[:, :, 0:1]
        inputs["latents"] = latents

        training_target = self.scheduler.training_target(input_latents, noise, timestep)
        noise_pred = self.model_fn(**inputs, timestep=timestep)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="none")
        if inputs.get("fuse_vae_embedding_in_latents", False):
            print(f'Applying loss mask for fused VAE embedding')
            loss_mask = torch.ones_like(loss)
            loss_mask[:, :, 0:1] = 0
            loss = (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1)
        else:
            loss = loss.mean()
        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    def _install_wan22_codex_units(self):
        self.units = [
            WanVideoUnit_Wan22CodexVACE144() if isinstance(unit, _WanVideoUnit_VACE_raymap) else unit
            for unit in self.units
            # 把官方的VACE单元替换成Wan2.2 Codex版本，后者构建144-channel的VACE输入
        ]

    @staticmethod
    def from_pretrained(*args, **kwargs):
        pipe = _WanVideoPipelineBase.from_pretrained(*args, **kwargs)
        pipe.units = [
            WanVideoUnit_Wan22CodexVACE144() if isinstance(unit, _WanVideoUnit_VACE_raymap) else unit
            for unit in pipe.units
        ]
        pipe.training_loss = types.MethodType(WanVideoPipeline.training_loss, pipe)
        return pipe
