import os

import diffusers
import torch
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import register_to_config
from huggingface_hub import utils
from packaging.version import Version
from torch import nn

from .utils import NunchakuModelLoaderMixin, pad_tensor
from .._C import QuantizedFluxModel, utils as cutils
from ..utils import fetch_or_download
from .FB_cache import *

SVD_RANK = 32
num_transformer_blocks = 19
num_single_transformer_blocks = 38


class NunchakuFluxTransformerBlocks(nn.Module):
    def __init__(
        self,
        m,
        device: str | torch.device,
        residual_diff_threshold: float = 0.06,
        return_hidden_states_first: bool = False,
        return_hidden_states_only: bool = False,
    ):
        super().__init__()
        self.m = m
        self.dtype = torch.bfloat16
        self.device = device
        self.residual_diff_threshold = residual_diff_threshold
        self.return_hidden_states_first = return_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only

    def call_remaining_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
    ):
        original_hidden_states = hidden_states.clone()  # for final residual
        original_encoder_hidden_states = encoder_hidden_states.clone()

        for idx in range(1, num_transformer_blocks):
            updated_h, updated_enc = self.m.forward_layer(
                idx,
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,   # image portion
                rotary_emb_txt,   # text portion
            )
            hidden_states = updated_h
            encoder_hidden_states = updated_enc

        cat_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for idx in range(num_single_transformer_blocks):
            cat_hidden_states = self.m.forward_single_layer(
                idx,
                cat_hidden_states,
                temb,
                rotary_emb_single,
            )

        final_encoder_hidden_states = cat_hidden_states[:, : encoder_hidden_states.shape[1], ...]
        final_hidden_states = cat_hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states_residual = final_hidden_states - original_hidden_states
        encoder_hidden_states_residual = final_encoder_hidden_states - original_encoder_hidden_states

        return final_hidden_states, final_encoder_hidden_states, hidden_states_residual, encoder_hidden_states_residual

    def forward(
        self,
        /,
        hidden_states: torch.Tensor,      # [batch, img_tokens, dim]
        temb: torch.Tensor,               # [batch, dim]
        encoder_hidden_states: torch.Tensor,  # [batch, txt_tokens, dim]
        image_rotary_emb: torch.Tensor,   # shape [1, 1, batch*(txt+img), head_dim/2, 1, 2]
        joint_attention_kwargs=None,
    ):
        batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(self.device)
        temb = temb.to(self.dtype).to(self.device)
        image_rotary_emb = image_rotary_emb.to(self.device)

        # Prepare rotary embeddings
        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == batch_size * (txt_tokens + img_tokens)

        image_rotary_emb = image_rotary_emb.reshape(
            [batch_size, (txt_tokens + img_tokens)] + list(image_rotary_emb.shape[3:])
        )

        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]
        rotary_emb_single = image_rotary_emb  # entire sequence

        rotary_emb_txt = pad_tensor(rotary_emb_txt, 256, dim=1)
        rotary_emb_img = pad_tensor(rotary_emb_img, 256, dim=1)
        rotary_emb_single = pad_tensor(rotary_emb_single, 256, dim=1)

        if self.residual_diff_threshold <= 0.0:
            for idx in range(num_transformer_blocks):
                updated_h, updated_enc = self.m.forward_layer(
                    idx,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    rotary_emb_img,
                    rotary_emb_txt,
                )
                hidden_states = updated_h
                encoder_hidden_states = updated_enc

            cat_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for idx in range(num_single_transformer_blocks):
                cat_hidden_states = self.m.forward_single_layer(
                    idx,
                    cat_hidden_states,
                    temb,
                    rotary_emb_single,
                )

            encoder_hidden_states = cat_hidden_states[:, :txt_tokens, ...]
            hidden_states = cat_hidden_states[:, txt_tokens:, ...]

        else:
            original_hidden_states = hidden_states

            first_updated_h, first_updated_enc = self.m.forward_layer(
                0,
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,
                rotary_emb_txt,
            )

            hidden_states = first_updated_h
            encoder_hidden_states = first_updated_enc

            first_hidden_states_residual = hidden_states - original_hidden_states
            del original_hidden_states

            parallelized = False 
            can_use_cache = get_can_use_cache(
                first_hidden_states_residual,
                threshold=self.residual_diff_threshold,
                parallelized=parallelized
            )

            torch._dynamo.graph_break()

            if can_use_cache:
                del first_hidden_states_residual
                hidden_states, encoder_hidden_states = apply_prev_hidden_states_residual(
                    hidden_states, encoder_hidden_states
                )
            else:
                set_buffer("first_hidden_states_residual", first_hidden_states_residual)
                del first_hidden_states_residual

                (
                    hidden_states,
                    encoder_hidden_states,
                    hidden_states_residual,
                    encoder_hidden_states_residual,
                ) = self.call_remaining_transformer_blocks(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    rotary_emb_img,
                    rotary_emb_txt,
                    rotary_emb_single,
                )

                set_buffer("hidden_states_residual", hidden_states_residual)
                set_buffer("encoder_hidden_states_residual", encoder_hidden_states_residual)

            torch._dynamo.graph_break()

        encoder_hidden_states = encoder_hidden_states.to(original_dtype).to(original_device)
        hidden_states = hidden_states.to(original_dtype).to(original_device)

        if self.return_hidden_states_only:
            return hidden_states
        else:
            if self.return_hidden_states_first:
                return (hidden_states, encoder_hidden_states)
            else:
                return (encoder_hidden_states, hidden_states)


## copied from diffusers 0.30.3
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)

    USE_SINCOS = True
    if USE_SINCOS:
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 1, 2)
    else:
        out = out.view(batch_size, -1, dim // 2, 1, 1)

    # stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    # out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super(EmbedND, self).__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def load_quantized_module(path: str, device: str | torch.device = "cuda", use_fp4: bool = False) -> QuantizedFluxModel:
    device = torch.device(device)
    assert device.type == "cuda"
    m = QuantizedFluxModel()
    cutils.disable_memory_auto_release()
    m.init(use_fp4, True, 0 if device.index is None else device.index)
    m.load(path)
    return m


class NunchakuFluxTransformer2dModel(FluxTransformer2DModel, NunchakuModelLoaderMixin):
    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int] = (16, 56, 56),
    ):
        super(NunchakuFluxTransformer2dModel, self).__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=0,
            num_single_layers=0,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        device = kwargs.get("device", "cuda")
        precision = kwargs.get("precision", "int4")
        assert precision in ["int4", "fp4"]
        transformer, transformer_block_path = cls._build_model(pretrained_model_name_or_path, **kwargs)
        
        m = load_quantized_module(transformer_block_path, device=device, use_fp4=precision == "fp4")
        transformer.inject_quantized_module(m, device)
        return transformer, m


    def update_lora_params(self, path: str):
        path = fetch_or_download(path)
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxTransformerBlocks)
        block.m.load(path, True)

    def set_lora_strength(self, strength: float = 1):
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxTransformerBlocks)
        block.m.setLoraScale(SVD_RANK, strength)

    def inject_quantized_module(self, m: QuantizedFluxModel, device: str | torch.device = "cuda"):
        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=[16, 56, 56])

        ### Compatible with the original forward method
        self.transformer_blocks = nn.ModuleList([NunchakuFluxTransformerBlocks(m, device)])
        self.single_transformer_blocks = nn.ModuleList([])

        return self
    
    def set_residual_diff_threshold(self, threshold: float):
        """
        Sets the residual_diff_threshold on all NunchakuFluxTransformerBlocks in this model.
        """
        for block in self.transformer_blocks:
            if isinstance(block, NunchakuFluxTransformerBlocks):
                block.residual_diff_threshold = threshold

        for block in self.single_transformer_blocks:
            if isinstance(block, NunchakuFluxTransformerBlocks):
                block.residual_diff_threshold = threshold

    def get_residual_diff_threshold(self) -> float:
        """
        Returns the residual_diff_threshold used by the first NunchakuFluxTransformerBlocks,
        or 0.0 if none found.
        """
        for block in self.transformer_blocks:
            if isinstance(block, NunchakuFluxTransformerBlocks):
                return block.residual_diff_threshold

        return 0.0
