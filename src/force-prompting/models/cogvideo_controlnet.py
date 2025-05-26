from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from diffusers.models.transformers.cogvideox_transformer_3d import Transformer2DModelOutput, CogVideoXBlock
from diffusers.utils import is_torch_version
from diffusers.loaders import  PeftAdapterMixin
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed # CogVideoXPatchEmbed
from .cogvideo_patch_embed import CogVideoXPatchEmbed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor2_0
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero, AdaLayerNormZeroSingle
from diffusers.configuration_utils import ConfigMixin, register_to_config

# imported from https://github.com/lllyasviel/ControlNet/blob/ed85cd1e25a5ed592f7d8178495b4483de0331bf/cldm/cldm.py#L9
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class CogVideoXControlnet(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        vae_channels: int = 16,
        in_channels: int = 3,
        downscale_coef: int = 8,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        out_proj_dim = None,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )
            
        start_channels = in_channels * (downscale_coef ** 2)
        # Nate changed to 12 so we could add the output to the hidden instead of concatenate
        # input_channels = [start_channels, start_channels // 2, start_channels // 4]
        # input_channels = [start_channels, start_channels // 2, start_channels // 12] 
        # input_channels = [start_channels, start_channels // 2, 32] # THIS WAS FOR NATEs zero conv stuff
        input_channels = [start_channels, start_channels // 2, 32]
        self.input_channels = input_channels
        self.unshuffle = nn.PixelUnshuffle(downscale_coef)
        
        self.controlnet_encode_first = nn.Sequential(
            nn.Conv2d(input_channels[0], input_channels[1], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[1]),
            nn.ReLU(),
        )

        self.controlnet_encode_second = nn.Sequential(
            nn.Conv2d(input_channels[1], input_channels[2], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[2]),
            nn.ReLU(),
        )

        # Same number of input and output channels
        self.controlnet_zero_conv_before = zero_module(
            nn.Conv3d(in_channels=input_channels[-1], out_channels=input_channels[-1], kernel_size=1, stride=1, padding=0)
        )
        
        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=64,#input_channels[2],#vae_channels + input_channels[2],
            embed_dim=inner_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 3. Define spatio-temporal transformers blocks...
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        # ... as well as the zero-convs that we apply after
        self.controlnet_zero_convs_after = nn.ModuleList(
            [
                zero_module(
                    nn.Conv3d(in_channels=3072, out_channels=3072, kernel_size=1, stride=1, padding=0)
                )
                for _ in range(num_layers)
            ]
        )


        self.out_projectors = None
        if out_proj_dim is not None:
            self.out_projectors = nn.ModuleList(
                [nn.Linear(inner_dim, out_proj_dim) for _ in range(num_layers)]
            )
            
        self.gradient_checkpointing = False
        
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def compress_time(self, x, num_frames):
        x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
        batch_size, frames, channels, height, width = x.shape
        x = rearrange(x, 'b f c h w -> (b h w) c f')
        
        if x.shape[-1] % 2 == 1:
            x_first, x_rest = x[..., 0], x[..., 1:]
            if x_rest.shape[-1] > 0:
                x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

            x = torch.cat([x_first[..., None], x_rest], dim=-1)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = rearrange(x, '(b h w) c f -> (b f) c h w', b=batch_size, h=height, w=width)
        return x
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):

        batch_size, num_frames, channels, height, width = controlnet_states.shape # (1, 49, 3, 480, 720)
        # 0. Controlnet encoder
        controlnet_states = rearrange(controlnet_states, 'b f c h w -> (b f) c h w') # (1, 49, 3, 480, 720) --> (49, 3, 480, 720)
        controlnet_states = self.unshuffle(controlnet_states) # (49, 192, 60, 90)
        controlnet_states = self.controlnet_encode_first(controlnet_states) # (49, 96, 60, 90)
        controlnet_states = self.compress_time(controlnet_states, num_frames=num_frames) # (25, 96, 60, 90)
        num_frames = controlnet_states.shape[0] // batch_size # 25

        controlnet_states = self.controlnet_encode_second(controlnet_states) # (25, 32, 60, 90)
        controlnet_states = self.compress_time(controlnet_states, num_frames=num_frames) # (13, 32, 60, 90)
        controlnet_states = rearrange(controlnet_states, '(b f) c h w -> b f c h w', b=batch_size) # (1, 13, 32, 60, 90)

        # concatenate along the channel dimension
        hidden_states = torch.cat([hidden_states, controlnet_states], dim=2) # (1, 13, 64, 60, 90)

        """v3
        # ControlNet zero convolution; Nate implemented this...
        controlnet_states = rearrange(controlnet_states, 'b f c h w -> b c f h w') # (1, 13, 32, 60, 90) --> (1, 32, 13, 60, 90)
        controlnet_states = self.controlnet_zero_conv_before(controlnet_states) # (1, 32, 13, 60, 90)
        controlnet_states = rearrange(controlnet_states, 'b c f h w -> b f c h w') # (1, 32, 13, 60, 90) --> (1, 13, 32, 60, 90)
        # Nate note: the original implementation had "hidden_states = torch.cat([hidden_states, controlnet_states], dim=2)"
        # but in the controlnet paper, the ControlNet states need to be ADDED to the original signal, 
        # not concatenated. this ensures control flow is the same... so I replaced that with the following--
        hidden_states += controlnet_states # (1, 13, 32, 60, 90) --> (1, 13, 32, 60, 90) # WITH THIS!! AND CHANGED A DIMENSION IN THE patch_embed
        """

        """v2
        # maybe we should only have ONE channel's zero-conv? seems wasteful to have 32 different convs... 
        # WAIT NO... the hidden states 32 channels have different purposes, and we want the controlnet states to 
        # adapt to each of them independently. so this is fine as is with 32 independent channels!
        # import pdb; pdb.set_trace()
        controlnet_states = rearrange(controlnet_states, 'b f c h w -> b c f h w') # (1, 13, 32, 60, 90) --> (1, 32, 13, 60, 90)
        controlnet_states = self.controlnet_zero_conv_before(controlnet_states) # (1, 32, 13, 60, 90) --> (1, 32, 13, 60, 90)
        controlnet_states = rearrange(controlnet_states, 'b c f h w -> b f c h w') # (1, 32, 13, 60, 90) --> (1, 13, 32, 60, 90)
        hidden_states += controlnet_states # (1, 13, 32, 60, 90) --> (1, 13, 32, 60, 90) # WITH THIS!! AND CHANGED A DIMENSION IN THE patch_embed "
        """

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        # text_embed: (1, 226, 4096), image_embed: (1, 13, 64, 60, 90) --> (1, 17776, 3072)
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states) # (1, 17776, 3072)
        hidden_states = self.embedding_dropout(hidden_states) # (1, 17776, 3072)

        text_seq_length = encoder_hidden_states.shape[1] # 226
        encoder_hidden_states = hidden_states[:, :text_seq_length] # (1, 226, 3072)
        hidden_states = hidden_states[:, text_seq_length:] # (1, 17550, 3072)
        
        controlnet_hidden_states = ()
        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, # (1, 17550, 1920), where 17550 = 13 * 90 * 15 = frames * width 
                    encoder_hidden_states, # (1, 226, 1920), where 226 = text_seq_length
                    emb, # (1, 512)
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
                
            if self.out_projectors is not None:
                controlnet_hidden_states += (self.out_projectors[i](hidden_states),)
            else:
                controlnet_hidden_states += (hidden_states,)
            
        if not return_dict:
            DO_ZERO_CONV = True
            # need to pass them through the zero conv, I think...
            controlnet_hidden_states_after_zero_conv = ()
            for i, (zero_conv, hidden_states) in enumerate(zip(self.controlnet_zero_convs_after, controlnet_hidden_states)):
                hidden_states = hidden_states.unsqueeze(-1).unsqueeze(-1) # (1, 17550, 1920) --> (1, 17550, 1920, 1, 1)
                hidden_states = rearrange(hidden_states, 'b s c h w -> b c s h w') # (1, 17550, 1920, 1, 1) --> (1, 1920, 17550, 1, 1)
                if DO_ZERO_CONV:
                    hidden_states = zero_conv(hidden_states) # (1, 1920, 17550, 1, 1) --> (1, 1920, 17550, 1, 1)
                hidden_states = rearrange(hidden_states, 'b c s h w -> b s c h w') # (1, 1920, 17550, 1, 1) --> (1, 17550, 1920, 1, 1)
                hidden_states = hidden_states.squeeze(-1).squeeze(-1) # (1, 17550, 1920, 1, 1) --> (1, 17550, 1920)
                controlnet_hidden_states_after_zero_conv += (hidden_states,)
            return (controlnet_hidden_states_after_zero_conv,)
        else:
            raise NotImplementedError
            return Transformer2DModelOutput(sample=controlnet_hidden_states)