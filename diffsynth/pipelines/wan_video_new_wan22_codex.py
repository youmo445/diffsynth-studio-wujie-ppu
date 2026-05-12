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
        ray_o_latents = self._as_latents(pipe, ray_map_o, "ray_map_o", tiled, tile_size, tile_stride)
        ray_d_latents = self._as_latents(pipe, ray_map_d, "ray_map_d", tiled, tile_size, tile_stride)
        action_latents = self._as_latents(pipe, action_source, "action_map/vace_video", tiled, tile_size, tile_stride)

        shapes = {tuple(u.shape[2:]) for u in (ray_o_latents, ray_d_latents, action_latents)}
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

    def _install_wan22_codex_units(self):
        self.units = [
            WanVideoUnit_Wan22CodexVACE144() if isinstance(unit, _WanVideoUnit_VACE_raymap) else unit
            for unit in self.units
        ]

    @staticmethod
    def from_pretrained(*args, **kwargs):
        pipe = _WanVideoPipelineBase.from_pretrained(*args, **kwargs)
        pipe.units = [
            WanVideoUnit_Wan22CodexVACE144() if isinstance(unit, _WanVideoUnit_VACE_raymap) else unit
            for unit in pipe.units
        ]
        return pipe
