import torch
from .wan_video_dit import DiTBlock
from .utils import hash_state_dict_keys

class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs):
        # import pdb;pdb.set_trace()
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
        enable_global_cross_attn=False,
        global_context_dim=16,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.dim = dim
        self.num_heads = num_heads
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.vace_global_enabled = False
        if enable_global_cross_attn:
            self.enable_global_cross_attn(global_context_dim=global_context_dim)

    def enable_global_cross_attn(self, global_context_dim=16):
        if self.vace_global_enabled:
            return
        ref = next(self.parameters())
        self.vace_global_proj = torch.nn.Linear(global_context_dim, self.dim)
        self.vace_global_norm_q = torch.nn.LayerNorm(self.dim)
        self.vace_global_norm_kv = torch.nn.LayerNorm(self.dim)
        self.vace_global_attn = torch.nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.vace_global_gate = torch.nn.Parameter(torch.zeros(()))
        self.vace_global_enabled = True
        self.vace_global_proj.to(device=ref.device, dtype=ref.dtype)
        self.vace_global_norm_q.to(device=ref.device, dtype=ref.dtype)
        self.vace_global_norm_kv.to(device=ref.device, dtype=ref.dtype)
        self.vace_global_attn.to(device=ref.device, dtype=ref.dtype)
        self.vace_global_gate.data = self.vace_global_gate.data.to(device=ref.device, dtype=ref.dtype)

    def forward(
        self, x, vace_context, context, t_mod, freqs, vace_global_context=None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        if vace_global_context is not None:
            if not self.vace_global_enabled:
                raise RuntimeError("vace_global_context was provided, but VaceWanModel global cross-attn is not enabled.")
            g = vace_global_context.to(device=c.device, dtype=c.dtype)
            if g.ndim == 2:
                g = g.unsqueeze(0)
            if g.shape[0] == 1 and c.shape[0] > 1:
                g = g.repeat(c.shape[0], 1, 1)
            elif g.shape[0] != c.shape[0]:
                raise ValueError(f"Global context batch {g.shape[0]} does not match VACE batch {c.shape[0]}.")
            g = self.vace_global_proj(g)
            delta, _ = self.vace_global_attn(
                self.vace_global_norm_q(c),
                self.vace_global_norm_kv(g),
                self.vace_global_norm_kv(g),
                need_weights=False,
            )
            c = c + self.vace_global_gate * delta

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block in self.vace_blocks:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c, x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c, x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                c = block(c, x, context, t_mod, freqs)
        # import pdb;pdb.set_trace()
        hints = torch.unbind(c)[:-1]
        return hints
    
    @staticmethod
    def state_dict_converter():
        return VaceWanModelDictConverter()
    
    
class VaceWanModelDictConverter:
    def __init__(self):
        pass
    
    def from_civitai(self, state_dict):
        state_dict_ = {name: param for name, param in state_dict.items() if name.startswith("vace")}
        if hash_state_dict_keys(state_dict_) == '3b2726384e4f64837bdf216eea3f310d': # vace 14B
            config = {
                "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35),
                "vace_in_dim": 96,
                "patch_size": (1, 2, 2),
                "has_image_input": False,
                "dim": 5120,
                "num_heads": 40,
                "ffn_dim": 13824,
                "eps": 1e-06,                
            }
        else:
            config = {}
        if any(name.startswith("vace_global_") for name in state_dict_):
            config["enable_global_cross_attn"] = True
        return state_dict_, config
