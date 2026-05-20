import torch

from .wan_video_vace import VaceWanAttentionBlock


class VaceWan22CodexModel(torch.nn.Module):
    """Minimal Wan2.2-TI2V-5B VACE branch for 144-channel latent controls.

    The intended control tensor is channel-concatenated
    [ray_o_latents, ray_d_latents, action_map_latents], where each part is
    encoded by Wan2.2 VAE and has 48 channels.
    """

    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=144,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=3072,
        num_heads=24,
        ffn_dim=14336,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = tuple(vace_layers)
        self.vace_in_dim = vace_in_dim
        self.dim = dim
        self.num_heads = num_heads
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        self.vace_blocks = torch.nn.ModuleList(
            [
                VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
                for i in self.vace_layers
            ]
        )
        self.vace_patch_embedding = torch.nn.Conv3d(
            vace_in_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def init_from_dit(self, dit):
        """Copy matching transformer block weights from the Wan2.2 DiT."""
        summaries = []
        for layer_id, vace_block in zip(self.vace_layers, self.vace_blocks):
            if layer_id >= len(dit.blocks):
                raise ValueError(f"VACE layer {layer_id} exceeds DiT block count {len(dit.blocks)}.")
            load_result = vace_block.load_state_dict(dit.blocks[layer_id].state_dict(), strict=False)
            summaries.append(
                {
                    "layer": layer_id,
                    "missing": list(load_result.missing_keys),
                    "unexpected": list(load_result.unexpected_keys),
                }
            )
        return summaries

    @staticmethod
    def state_dict_converter():
        return VaceWan22CodexModelDictConverter()

    def forward(
        self,
        x,
        vace_context,
        context,
        t_mod,
        freqs,
        vace_global_context=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        if vace_global_context is not None:
            raise ValueError("VaceWan22CodexModel does not use global reference cross-attention.")
        if vace_context.shape[1] != self.vace_in_dim:
            raise ValueError(
                f"Expected vace_context with {self.vace_in_dim} channels, got {vace_context.shape[1]}."
            )
        # vace_context: [1, 144, 5, 30, 14]
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        # [[1, 3072, 5, 15, 7]]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        # [[1, 525 = 5 * 15 * 7, 3072]]
        c = torch.cat(
            [
                torch.cat(
                    [u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                    dim=1,
                )
                for u in c
            ]
        )

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block in self.vace_blocks:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c,
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c,
                    x,
                    context,
                    t_mod,
                    freqs,
                    use_reentrant=False,
                )
            else:
                c = block(c, x, context, t_mod, freqs)
        return torch.unbind(c)[:-1]
        # 得到每个VACE block的输出，形状都是 [1, 525, 3072]，对应每个块的提示向量

class VaceWan22CodexModelDictConverter:
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if name.startswith("vace")}
        return state_dict, {}

    def from_diffusers(self, state_dict):
        return self.from_civitai(state_dict)
