import time
import copy
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import WanPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

from src.pipelines.base_pipeline import DualParalPipelineBaseWrapper
from src.distribution_utils import DistController, RuntimeConfig

class DualParal_WanAttnProcessor:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        cache = None,
        latent_num = None,
        select = None,
        select_all = None,
        out_dim = None
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        use_cache = False
        cache2, cache3 = None, None
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if latent_num is not None:
            cache2, cache3 = key[:, -latent_num:], value[:, -latent_num:]
            cutoff = latent_num//select_all*select
            cache2, cache3 = cache2[:, :cutoff].clone(), cache3[:, :cutoff].clone()
            
        if cache is not None and select>0:
            use_cache = True
            k_, v_ = cache[0], cache[1]
            key = torch.cat((key, k_), dim=1)
            value = torch.cat((value, v_), dim=1)
            cache = None
        if cache2 is not None and select>0:
            cache = (cache2, cache3)
        else: 
            cache = None
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)
            query = apply_rotary_emb(query, rotary_emb[0])
            key = apply_rotary_emb(key, rotary_emb[1])

        L, S = query.size(-2), key.size(-2)

        attn_mask = None
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        if latent_num is not None:
            hidden_states = hidden_states[:, :out_dim] #delete cache
            
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, cache

class DualParal_WanTransformerBlock(nn.Module):
    def __init__(
        self,
        block: WanTransformerBlock,
    ):
        super().__init__()
        self.block = block

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        cache: Optional[Tuple] = None,
        latent_num = None,
        select = None,
        select_all = None,
        attention_mask = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.block.scale_shift_table + temb.float()
        ).chunk(6, dim=1)
        # 1. Self-attention
        norm_hidden_states = (self.block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output, cache = self.block.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, cache=cache, \
                                            latent_num=latent_num, select=select, select_all=select_all, attention_mask=attention_mask, out_dim=hidden_states.size(1))
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.block.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output, _ = self.block.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, \
                                            latent_num=None, select=select, select_all=select_all, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.block.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.block.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states, cache

class DualParalWanPipeline(DualParalPipelineBaseWrapper):
    def __init__(
        self,
        model: WanPipeline,
        parallel_config: DistController,
        runtimeconfig: RuntimeConfig,
    ):
        DualParalPipelineBaseWrapper.__init__(self, parallel_config, runtimeconfig)
        device, dtype = self.parallel_config.device, self.runtime_config.dtype

        self.tokenizer = model.tokenizer
        self.text_encoder = model.text_encoder
        self.transformer = model.transformer.blocks # Only Dit Blocks 
        self.vae = model.vae
        self.scheduler = model.scheduler
        self.scheduler_dict = {}
        self.model = model   # Other function or property all in self.model
        self.cache = []
        self.attention_mask = []
        self.tmp = None

        pretransformer, finaltransformer = False, False
        if self.parallel_config.world_size==1:
            pretransformer, finaltransformer = True, True
        elif self.parallel_config.rank==0:
            pretransformer, finaltransformer = True, False
        elif self.parallel_config.rank>0 and self.parallel_config.rank<self.parallel_config.world_size-1:
            pretransformer, finaltransformer = False, False
        else:
            pretransformer, finaltransformer = False, True

        self.pretransformer, self.finaltransformer = pretransformer, finaltransformer
        self._split_transformer_backbone()
        del self.model.transformer.blocks
        self.model.transformer.blocks = self.transformer
        for idx, block in enumerate(self.transformer):
            block.attn1.set_processor(DualParal_WanAttnProcessor())
            block.attn2.set_processor(DualParal_WanAttnProcessor())
            self.transformer[idx] = DualParal_WanTransformerBlock(block)
    
    @torch.no_grad()  
    def onestep(self, 
        blocklatents, 
        latent_size = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        latent_num = None,
        select = 0,
        select_all = 0,
        verbose=False,
    ):
        if verbose: start_time = time.time()
        i = blocklatents.denoise_time
        z, t = blocklatents.z, self.timesteps[i]

        if verbose: print(f"[{self.parallel_config.device}] latents size {latent_size}, with cache {len(self.cache)}, with latent_num {latent_num}, with select_all {select_all}, with select {select}, with time {time.time()-start_time}")
        t = t.expand(1)
        t = torch.cat([t, t], 0)
        t = t.to(self.parallel_config.device, self.runtime_config.dtype).contiguous()

        latent_size = torch.Size([2] + list(latent_size)[1:])
        batch_size, num_channels, num_frames, height, width = latent_size
        p_t, p_h, p_w = self.model.transformer.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        self.tmp = torch.empty(*latent_size).to(self.parallel_config.device, self.runtime_config.dtype)
        rotary_emb_1 = self.model.transformer.rope(self.tmp)

        if len(self.cache)>0:
            # for position
            num_frames += select
            latent_size = torch.Size([batch_size, num_channels, num_frames, height, width])
        
        self.tmp = torch.empty(*latent_size).to(self.parallel_config.device, self.runtime_config.dtype)
        rotary_emb_2 = self.model.transformer.rope(self.tmp)
        rotary_emb = (rotary_emb_1, rotary_emb_2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.model.transformer.condition_embedder(
            t, self.encoder_hidden_states, self.encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

     
        if self.pretransformer:
            z = torch.cat([z, z], 0) 
            z = self.model.transformer.patch_embedding(z)
            z = z.flatten(2).transpose(1, 2)

        hidden_states = z.contiguous()
        use_cache_ = (len(self.cache) > 0)
        for now_idx, block in enumerate(self.transformer):
            cache = None if not use_cache_ else self.cache[now_idx]
            size_tmp = hidden_states.size()
            hidden_states, cache = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, cache=cache,\
                                        latent_num=latent_num, select=select, select_all=select_all, attention_mask=self.attention_mask)
            if not use_cache_:
                self.cache.append(cache)
            else:
                self.cache[now_idx] = cache

        if self.finaltransformer:
            shift, scale = (self.model.transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)
            hidden_states = hidden_states.contiguous()
            hidden_states = (self.model.transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
            hidden_states = self.model.transformer.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(
                batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
            )
            hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
            noise_pred = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
            noise_pred = noise_pred
            noise_pred, noise_uncond = noise_pred[0], noise_pred[1]
            noise_pred = noise_uncond + self.guidance_scale * (noise_pred - noise_uncond)
            noise_pred = noise_pred.unsqueeze(0)
            if verbose: print(f"[{self.parallel_config.device}] Final (use cache {use_cache_}) tiem {time.time()-start_time:.6f}s")
            return noise_pred
        else: 
            if verbose: print(f"[{self.parallel_config.device}] Final (use cache {use_cache_}) tiem {time.time()-start_time:.6f}s")
            return hidden_states 

    def get_model_args(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        do_classifier_free_guidance: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
    ):
        device, dtype = self.parallel_config.device, self.runtime_config.dtype
        self.guidance_scale = guidance_scale

        # 1. Get Prompt Embedding
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        prompt_embeds = prompt_embeds.to(device, dtype, non_blocking=True)
        negative_prompt_embeds = negative_prompt_embeds.to(device, dtype, non_blocking=True) if negative_prompt_embeds is not None else None
        if do_classifier_free_guidance:
            self.encoder_hidden_states = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        else:
            self.encoder_hidden_states = prompt_embeds
        self.encoder_hidden_states_image = None
        
        del self.text_encoder
        del self.tokenizer
        del self.model.text_encoder
        del self.model.tokenizer
        torch.cuda.empty_cache()

        # 2. Prepare Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.timesteps = self.scheduler.timesteps

    def to_device(self, device, dtype, **kwargs):
        self.model = self.model.to(device, dtype, **kwargs) if self.model is not None else None
        self.vae = self.vae.to(dtype)
        return self
    
    def add_scheduler(self, idx):
        self.scheduler_dict[idx] = copy.deepcopy(self.scheduler)
    
    def del_scheduler(self, idx):
        del self.scheduler_dict[idx] 

    def get_scheduler_dict(self, idx):
        return self.scheduler_dict[idx]

    @torch.no_grad()
    def get_video(
        self, 
        latents, 
        output_type: Optional[str] = "np",
        verbose=False):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype, non_blocking=True)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.model.video_processor.postprocess_video(video, output_type=output_type)
        return video