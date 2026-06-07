import types

import torch

from .wan_video_new import *  # noqa: F401,F403
from .wan_video_new import (
    WanVideoPipeline as _WanVideoPipelineBase,
    WanVideoUnit_VACE_raymap as _WanVideoUnit_VACE_raymap,
)
from ..utils import PipelineUnit


class WanVideoUnit_Wan22CodexVACE144(PipelineUnit):
    """Build 144-channel Wan2.2 VACE controls from ray/action maps.

    This unit intentionally does not append a mask. It encodes:
    ray_map_o -> 48 channels, ray_map_d -> 48 channels, action_map/vace_video
    -> 48 channels, then concatenates them into vace_context.
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
            raise NotImplementedError("暂时不支持直接输入ray/action latent tensor，必须输入原始视频/图像/动作图并通过VAE编码成latent。")
            return latents
        video = pipe.preprocess_video(value)
        return pipe.vae.encode(
            video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

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
        ray_o_latents = self._as_latents(pipe, ray_map_o, "ray_map_o", tiled, tile_size, tile_stride) # [1, 48, T', H', W']
        ray_d_latents = self._as_latents(pipe, ray_map_d, "ray_map_d", tiled, tile_size, tile_stride)
        action_latents = self._as_latents(pipe, action_source, "action_map/vace_video", tiled, tile_size, tile_stride)
        shapes = {tuple(u.shape[2:]) for u in (ray_o_latents, ray_d_latents, action_latents)}
        # {}会去重，如果去重后不止一个shape，说明ray/action latent的时间/空间维度不匹配，无法拼接成VACE输入了
        if len(shapes) != 1:
            raise ValueError(
                "ray/action latent temporal-spatial shapes must match, got "
                f"{ray_o_latents.shape}, {ray_d_latents.shape}, {action_latents.shape}."
            )
        z_dim = getattr(pipe.vae, "z_dim", getattr(pipe.vae.model, "z_dim", 48))
        for name, latents in (
            ("ray_map_o", ray_o_latents),
            ("ray_map_d", ray_d_latents),
            ("action_map/vace_video", action_latents),
        ):
            if latents.shape[1] != z_dim:
                raise ValueError(f"{name} should have {z_dim} channels after VAE encode, got {latents.shape[1]}.")

        vace_context = torch.cat((ray_o_latents, ray_d_latents, action_latents), dim=1)
        expected_in_dim = getattr(pipe.vace, "vace_in_dim", 144)
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
