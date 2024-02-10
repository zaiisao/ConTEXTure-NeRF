import einops
import torch
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context) # JA: In our experiment context has a batch size of 8 and h a size of 5
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None: # JA: If control is not none, add the ControlNet output (at middle block) to the output of the middle block in the UNet
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1) # JA: If the control is not None, get the ControlNet output of the current block and add it to the output of the stable diffusion UNet
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # JA: From https://arxiv.org/pdf/2302.05543.pdf
        # In particular, we use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides (activated by ReLU, using 16, 32, 64, 128,
        # channels respectively, initialized with Gaussian weights and trained jointly with the full model) to encode an image-space condition c_i into a
        # feature space conditioning vector c_f as, c_f = E(ci). (4) The conditioning vector c_f is passed into the ControlNet.
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, #use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, #use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context) # JA: input_input_block is the tiny encoder of the controlnet

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config) # JA: control_model is an object of the ControlNet class
        self.control_key = control_key # JA: control_key is "hint"
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs): # JA: Override get_input of LatentDiffusion (Zero123 version)
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs) # JA: get_input of ControlLDM invokes get_input of LatentDiffusion. x is a tensor (b, 4, 32, 32), c is a dict with c_crossattn (4, 1, 768) and c_concat in the latent space (4, 4, 32, 32)
        control = batch[self.control_key] # JA: control_key is hint
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w') # JA: (4, 256, 256, 3) -> (4, 3, 256, 256) (in the pixel space). x is in the latent space
        control = control.to(memory_format=torch.contiguous_format).float()

        # JA: c = { c_concat: x } or { c_concat: x, c_crossattn: y } or { c_crossattn: y }
        # JA: If we want to use both the concat condition and the hint condition, we have to make the
        # distinction here, resulting in three values in the dictionary to be returned.
        c['c_control'] = [control] # JA: control has a shape of (4, 3, 256, 256)
        return x, c

    #MJ: called by p_losses() which is called by shared_step() which is called by training_step()
    def apply_model(self, x_noisy, t, cond, *args, **kwargs): # JA: Override apply_model method of LatentDiffusion; MJ: cond could be uncond
        assert isinstance(cond, dict) # JA: Here, cond contains cond['c_control']
        diffusion_model = self.model.diffusion_model # JA: diffusion_model is the ControlledUnetModel

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        # Added by JA: The default ControlNet code passes the hint into the control model as c_concat, even though
        # it is not a true concat condition. This becomes a problem when using ControlNet to fine-tune a model that
        # already uses c_concat.

        # if cond_hint is None: # JA: This means there is no control
        #     eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        # else:
        if self.model.conditioning_key in ['hybrid', 'concat', 'hybrid-adm']:
            # JA: If c_control does not exist, then it is assumed that c_concat is the hint. If both c_control and
            # c_concat both exist, then c_concat is to be concatenated in the normal manner, with x_noisy which
            # is seen in the DiffusionWrapper.
            x = torch.cat([x_noisy] + cond['c_concat'], dim=1) # JA: In this case, the channels of x is 8
        elif self.model.conditioning_key == 'crossattn':
            # JA: Set x to be the original value, which is x_noisy. This is the default behavior in ControlNet
            x = x_noisy # JA: In this case, the channel is 4
        else:
            raise NotImplementedError

        c_control = cond['c_control'] if 'c_control' in cond else None
        if c_control is not None:
            control = self.control_model(x=x, hint=torch.cat(cond['c_control'], 1), timesteps=t, context=cond_txt) # JA: the control is the skip connections from the encoding blocks of copied stable diffusion
            control = [c * scale for c, scale in zip(control, self.control_scales)]
        else:
            control = None

        eps = diffusion_model(x=x, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps
    
    #MJ: In controlNet + zero123, we do not override  get_unconditional_conditioning defined in the zero123 LatentDiffusion
    
    # @torch.no_grad()
    # def get_unconditional_conditioning(self, N):
    #     return self.get_learned_conditioning([""] * N)

    #MJ: called by on_train_batch_end() which is called automatically at the end of training_step()
    ##MJ: log_imgages() of ControlLDM  different from that of zero123 (which in turn a little bit different from LatentDiffusion)
    @torch.no_grad() 
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        
        #MJ: c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]; original in ControlLDM
        c_con, c_cat, c_cross = c["c_control"][0][:N], c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        #MJ: LDM uses c["c_concat"] to store the control image, but we use it to store the real concat condition,
        # uses c["c_control"] to store the control image
        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        
        #MJ: log["control"] = c_cat * 2.0 - 1.0 #MJ: [0,1] => [-1,1]; new in ControlLDM
        log["control"] = c_con * 2.0 - 1.0 # JA: ([-1, 1] * 2) - 1 => [-2, 2] - 1 => [-3, 1]
        #MJ: To run the neuralnet in sampling mode, we need to normalize the control image, because
        # it is not yet normalized.

        # import torchvision
        # for idx in range(N):
        #     torchvision.utils.save_image(log["reconstruction"][:N][idx], f"image_target_{idx}.png")
        #     torchvision.utils.save_image(batch['image_cond'][:N][idx].permute(2, 0, 1), f"image_cond_{idx}.png")
        #     torchvision.utils.save_image(log["control"][idx], f"hint_{idx}.png")
        #     torchvision.utils.save_image(
        #         log_txt_as_img((512, 512), str(batch['T'][:N][idx]), size=16),
        #         f"rotation_{idx}.png"
        #     )
        
        if type(batch[self.cond_stage_key]) == "str":   # If check added by JA
                                                        # Zero123 does not use text prompts, so we cannot use log_text_as_img
            log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample: #MJ: true in the case of inference stage
            # get denoise row
            
            # samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
            #                                          batch_size=N, ddim=use_ddim,
            #                                          ddim_steps=ddim_steps, eta=ddim_eta)
            #MJ:
            #In our controlNet+ zero123, we use both cond['c_concat'] and cond['c_control']
            cond = {"c_control": [c_con], "c_concat": [c_cat],"c_crossattn": [c_cross]}
                                                      
            samples, z_denoise_row = self.sample_log(cond=cond,
                                                      batch_size=N, ddim=use_ddim,
                                                      ddim_steps=ddim_steps, eta=ddim_eta)
            #MJ: because  unconditional_conditioning=None (which is the case by default) 
            #    => The pure conditional sample is performed, that is, without using the unconditional sampling direction
             
            
            
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            h, w = c_con.shape[-2], c_con.shape[-1]
            #MJ: copied from the zero123 LatentDiffusion:
            unconditional_guidance_label = [torch.zeros(N, 4, h // 8, w // 8).to(c_con.device)]
            # MJ: calls get_unconditonal_conditinong() defined in the zero123 LatentDiffusion 
            # JA: uc_full contains c_concat and c_crossattn conditions
            uc_full = self.get_unconditional_conditioning(N, unconditional_guidance_label, image_size=c_con.shape[-1])
            uc_full["c_control"] = [c_con] * N # JA: In this case, the c_control value of the uncond is not None. Therefore we do not need the hacked code of apply_model method. This is the case during the training but during the inference we need to use the hacked code of apply_model

            #MJ: In our controlNet+ zero123, we use both cond['c_concat'] and cond['c_control']
            
            cond = {
                "c_control": [c_con],
                "c_concat": [c_cat],
                "c_crossattn": [c_cross]
            }
                                                      
            # uc_cross = self.get_unconditional_conditioning(N) # JA: In our experiment, uc_cross shape is (1, 1, 768). It should be (4, 1, 768)
            # uc_cat = c_cat  # torch.zeros_like(c_cat) # 
            #MJ: We use cond['c_control'] to store the control image, and cond['c_concat'] to store the real concat condition
            
            # uc_con = c_con 
            # uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]} # original in ControlLDM
         
            # samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
            #                                  batch_size=N, ddim=use_ddim,
            #                                  ddim_steps=ddim_steps, eta=ddim_eta,
            #                                  unconditional_guidance_scale=unconditional_guidance_scale,
            #                                  unconditional_conditioning=uc_full,
            #                                  )
            
            samples_cfg, _ = self.sample_log(cond=cond, #MJ: cond contains all of the conditions, c_control, c_concat, c_crossattn
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,  
                                              #MJ: the unconditional conditioning used for "classifier-free guidance" contains the "unconditional conditions" for only c_concat and c_crossattn
                                             ) 
                                #MJ: self.sample_log() invokes ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
                                #which via some steps  invokes  self.model.apply_model(x, t, c), which is defined in ControlLDM,
                                # where c contains "c_control", "c_concat", "c_crossattn" conditions. 
            
            
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self) # JA: The sample_log function executes the inference (sampling) of the current network being trained
        
        #b, c, h, w = cond["c_concat"][0].shape; original in ControlLDM
        b, c, h, w = cond["c_control"][0].shape
        shape = (self.channels, h // 8, w // 8) 
                
        #MJ: controlLDM uses: shape = (self.channels, h // 8, w // 8), where h,w are obtained from
        # cond['c_concat'] which stores the control input in pixel space in controlLDM.
        
        #Also, in ControlLDM, cond["c_concat"] contains the control image in the pixel space, whose size =256;
        #It is because cond["c_concat"] is not used at all in the current ControlLDM implementations.
        #But, because zero123 uses the real "c_concat" condition, to be concatenated with the latent image,
        # we created a new field called "c_control" in cond dict and uses "c_concat" for the real concat condition.
        
        # Here zero123 uses:  shape = (self.channels, self.image_size, self.image_size), where self.image_size = 32, which is
        # set in the init of DDPM from the config file; This is the standard method, but the controlLDM author used a hack.
        # We follow this hackery of ControlLDM in order to prevent confusion.
        
        # shape = (self.channels, h, w) 
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        #MJ: When   unconditional_conditioning=None (which is the default) => The pure conditional sample is performed, that is, without using the unconditional sampling direction
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler:  # JA: This part is normally not in ControlLDM, but I restored it
                                # from the LatentDiffusion.configure_optimizers function
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler

        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
