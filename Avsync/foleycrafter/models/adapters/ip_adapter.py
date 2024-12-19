import torch
import torch.nn as nn
from diffusers import ControlNetModel
from foleycrafter.models.auffusion_unet import UNet2DConditionModel
import os

from foleycrafter.models.adapters.attention_processor import (
    IPAttnProcessor2_0,
    AttnProcessor2_0,
)

class AudioProjmodel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=512):
        super().__init__()
        mult = 4
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim * mult),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim * mult, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim),
        )
        #self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds)
        return clip_extra_context_tokens

class VideoProjmodel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=512):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(clip_embeddings_dim,cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    

class SytleNet(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, controlnet, audio_proj_model, ckpt_path=None):
        super().__init__()
        self.controlnet = controlnet
        self.audio_proj_model = audio_proj_model
        #self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, controlnet_image, audio_emeds):
        if len(audio_emeds.shape) == 2:
            # 添加序列维度
            audio_emeds = audio_emeds.unsqueeze(1) 
        audio_emb = self.audio_proj_model(audio_emeds)
        #encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                                    noisy_latents, 
                                    timesteps, 
                                    encoder_hidden_states = audio_emb,
                                    controlnet_cond = controlnet_image,
                                    return_dict = False
                                    )
                                    
        return down_block_res_samples,mid_block_res_sample

    def save_pretrained(self, save_directory):
        """保存模型到指定目录"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存 controlnet
        self.controlnet.save_pretrained(os.path.join(save_directory, "controlnet"))
        
        # 保存 audio_proj_model
        audio_proj_state = self.audio_proj_model.state_dict()
        torch.save(audio_proj_state, os.path.join(save_directory, "audio_proj_model.bin"))
    
    @classmethod
    def from_pretrained(cls, pretrained_path):
        """从保存的目录加载模型"""
        # 加载 controlnet
        controlnet = ControlNetModel.from_pretrained(
            os.path.join(pretrained_path, "controlnet")
        )
        
        # 创建并加载 audio_proj_model
        audio_proj_model = AudioProjmodel(
            cross_attention_dim=controlnet.config.cross_attention_dim,
            clip_embeddings_dim=512,
        )
        audio_proj_state = torch.load(os.path.join(pretrained_path, "audio_proj_model.bin"))
        audio_proj_model.load_state_dict(audio_proj_state)
        
        return cls(controlnet, audio_proj_model)


def trans_index(state_dict):
    new_state_dict = {}
    index_map = {
        1: 1,    # 第一个 attn2
        3: 5,    # 第二个 attn2
        5: 9,    # ...
        7: 13,
        9: 17,
        11: 21,
        13: 25,
        15: 29,
        17: 33,
        19: 37,
        21: 41,
        23: 45,
        25: 49,
        27: 53,
        29: 57,
        31: 61
    }
    if True:
        map2_to_map1 = {
                        1: 1,
                        4: 5,
                        7: 9,
                        10: 13,
                        13: 17,
                        16: 21,
                        19: 25,
                        22: 29,
                        25: 33, 
                        28: 37,
                        31: 41,
                        34: 45,
                        37: 49,
                        40: 53,
                        43: 57,
                        46: 61
                    }
    
    # 使用映射加载权重
    for old_idx, new_idx in map2_to_map1.items():
        new_state_dict[f"{new_idx}.to_k_ip.weight"] = state_dict["ip_adapter"][f"{old_idx}.to_k_ip.weight"]
        new_state_dict[f"{new_idx}.to_v_ip.weight"] = state_dict["ip_adapter"][f"{old_idx}.to_v_ip.weight"]
    
    return new_state_dict


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, unet, image_proj_model, adapter_modules, video_proj_model,ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.video_proj_model = video_proj_model

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, video_embeds,image_embeds,):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        encoder_hidden_states_2 = self.video_proj_model(video_embeds)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, 
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_hidden_states_2=encoder_hidden_states_2,).sample
        return noise_pred
    
    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        new_stat = trans_index(state_dict)
        self.adapter_modules.load_state_dict(new_stat, strict=True)
        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        
        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    
    def load_unet3(self,model_path):
        video_path = os.path.join(model_path,'video_proj_model.bin')
        unet3_path = os.path.join(model_path,'unet_attn3.bin')
        video_proj_state = torch.load(video_path,map_location='cpu')
        self.video_proj_model.load_state_dict(video_proj_state,strict=True)
        attn3_state_dict = torch.load(unet3_path,map_location='cpu')
        model_state_dict = self.unet.state_dict()
        model_state_dict.update(attn3_state_dict)
        self.unet.load_state_dict(model_state_dict)

    def save_pretrained(self, save_directory):
        """保存训练的参数到指定目录"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. 保存UNet中的attn3相关参数
        attn3_state_dict = {
            k: v for k, v in self.unet.state_dict().items()
            if ("attn3" in k) or ("norm3_0" in k)
        }
        torch.save(attn3_state_dict, os.path.join(save_directory, "unet_attn3.bin"))
        
        # 2. 保存audio_proj_model
        video_proj_state = self.video_proj_model.state_dict()
        torch.save(video_proj_state, os.path.join(save_directory, "video_proj_model.bin"))
        combined_state = {
        "image_proj": self.image_proj_model.state_dict(),
        "ip_adapter": self.adapter_modules.state_dict()
        }
        torch.save(combined_state, os.path.join(save_directory, "ip_adapter_plus.bin"))


        

    @classmethod
    def from_pretrained(cls, audio_base_pretrain, pretrained_path, ckpt_path):
        """从保存的目录加载模型
        
        Args:
            pretrained_path: 保存的训练参数路径
            base_unet_path: 基础UNet模型路径(如果不提供则使用pretrained_path)
        """
        # 1. 加载基础UNet

        unet = UNet2DConditionModel.from_pretrained(audio_base_pretrain, subfolder="unet", device_map=None, low_cpu_mem_usage=False)
        
        # 2. 加载训练过的attn3参数
        attn3_path = os.path.join(pretrained_path, "unet_attn3.bin")
        if os.path.exists(attn3_path):
            attn3_state = torch.load(attn3_path, map_location="cpu")
            # 更新UNet中的attn3参数
            current_state = unet.state_dict()
            for k, v in attn3_state.items():
                if k in current_state:
                    current_state[k].copy_(v)
        
        # 3. 创建并加载audio_proj_model
        audio_proj_model = AudioProjmodel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=512
        )
        audio_proj_state = torch.load(
            os.path.join(pretrained_path, "audio_proj_model.bin"),map_location="cpu"
        )
        audio_proj_model.load_state_dict(audio_proj_state)
        
        # 4. 创建并加载image_proj_model
        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=1024,
            clip_extra_context_tokens=4
        )
        
        """attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") or name.endswith("attn3.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
            else:
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size, 
                    cross_attention_dim=cross_attention_dim
                )
        
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())"""
        adapter_modules = None     
        # 6. 创建模型实例
        model = cls(
            unet=unet,
            image_proj_model=image_proj_model,
            adapter_modules=adapter_modules,
            audio_proj_model=audio_proj_model,
            #ckpt_path=ckpt_path
        )
        
        return model



class IPAdapter_audio(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, unet, image_proj_model, adapter_modules, video_proj_model,audio_proj_model,ckpt_path=None,model_path=None,encodec_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model ### for internvl 
        self.adapter_modules = adapter_modules ### ipa
        self.video_proj_model = video_proj_model #### 250frame image 
        self.audio_proj_model = audio_proj_model

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)
        
        if model_path is not None:
            self.load_unet3(model_path=model_path)
        
        if encodec_path is not None:
            self.load_encodec(encodec_path)



    def forward(self, noisy_latents, timesteps, encoder_hidden_states, video_embeds,image_embeds,prompt_audio):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        encoder_hidden_states_2 = self.video_proj_model(video_embeds)
        encoder_hidden_states_a = self.audio_proj_model(prompt_audio)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, 
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_hidden_states_2=encoder_hidden_states_2,
                                encoder_hidden_states_a=encoder_hidden_states_a
                                ).sample
        return noise_pred
    
    def load_unet3(self,model_path):
        video_path = os.path.join(model_path,'video_proj_model.bin')
        unet3_path = os.path.join(model_path,'unet_attn3.bin')
        video_proj_state = torch.load(video_path,map_location='cpu')
        self.video_proj_model.load_state_dict(video_proj_state,strict=True)
        attn3_state_dict = torch.load(unet3_path,map_location='cpu')
        model_state_dict = self.unet.state_dict()
        model_state_dict.update(attn3_state_dict)
        self.unet.load_state_dict(model_state_dict)
    

    def load_encodec(self,encodec_path):
        audio_path = os.path.join(encodec_path,'audio_proj_model.bin')
        uneta_path = os.path.join(encodec_path,'unet_attn_a.bin')
        audio_proj_state = torch.load(audio_path,map_location='cpu')
        self.audio_proj_model.load_state_dict(audio_proj_state,strict=True)
        attna_state_dict = torch.load(uneta_path,map_location='cpu')
        model_state_dict = self.unet.state_dict()
        model_state_dict.update(attna_state_dict)
        self.unet.load_state_dict(model_state_dict)
    
    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        ip_path = os.path.join(ckpt_path,'ip_adapter_plus.bin')

        state_dict = torch.load(ip_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        new_stat = trans_index(state_dict)
        self.adapter_modules.load_state_dict(new_stat, strict=True)
        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        
        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        #att_path = os.path.join(ckpt_path,'unet_attn3.bin')
        #att_dict = torch.load(att_path,map_location='cpu')
        #current_state = self.unet.state_dict()
        #current_state.update(att_dict)
        #self.unet.load_state_dict(current_state, strict=True)

        #video_path = os.path.join(ckpt_path,'video_proj_model.bin')
        #video_dict = torch.load(video_path,map_location='cpu')
        #self.video_proj_model.load_state_dict(video_dict, strict=True)

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


    def save_pretrained(self, save_directory):
        """保存训练的参数到指定目录"""
        os.makedirs(save_directory, exist_ok=True)


        attn3_state_dict = {
            k: v for k, v in self.unet.state_dict().items()
            if ("attn3" in k) or ("norm3_0" in k)
        }
        torch.save(attn3_state_dict, os.path.join(save_directory, "unet_attn3.bin"))
        
        # 2. 保存audio_proj_model
        video_proj_state = self.video_proj_model.state_dict()
        torch.save(video_proj_state, os.path.join(save_directory, "video_proj_model.bin"))
        combined_state = {
        "image_proj": self.image_proj_model.state_dict(),
        "ip_adapter": self.adapter_modules.state_dict()
        }
        torch.save(combined_state, os.path.join(save_directory, "ip_adapter_plus.bin"))
        
        # 1. 保存UNet中的attn3相关参数
        attna_state_dict = {
            k: v for k, v in self.unet.state_dict().items()
            if ("attn_a" in k) or ("norm_a" in k)
        }
        torch.save(attna_state_dict, os.path.join(save_directory, "unet_attn_a.bin"))
        
        # 2. 保存audio_proj_model
        audio_proj_state = self.audio_proj_model.state_dict()
        torch.save(audio_proj_state, os.path.join(save_directory, "audio_proj_model.bin"))

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def zero_initialize(module):
        for param in module.parameters():
            param.data.zero_()

    def zero_initialize_last_layer(module):
        last_layer = None
        for module_name, layer in module.named_modules():
            if isinstance(layer, torch.nn.Linear):
                last_layer = layer

        if last_layer is not None:
            last_layer.weight.data.zero_()
            last_layer.bias.data.zero_()

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim),
        )
        # zero initialize the last layer
        # self.zero_initialize_last_layer()

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class V2AMapperMLP(torch.nn.Module):
    def __init__(self, cross_attention_dim=512, clip_embeddings_dim=512, mult=4):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim * mult),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim * mult, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class TimeProjModel(torch.nn.Module):
    def __init__(self, positive_len, out_dim, feature_type="text-only", frame_nums: int = 64):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim

        self.position_dim = frame_nums

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        if feature_type == "text-only":
            self.linears = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, out_dim),
            )
            self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))

        elif feature_type == "text-image":
            self.linears_text = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, out_dim),
            )
            self.linears_image = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, out_dim),
            )
            self.null_text_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))
            self.null_image_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))

        # self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(
        self,
        boxes,
        masks,
        positive_embeddings=None,
    ):
        masks = masks.unsqueeze(-1)

        # # embedding position (it may includes padding as placeholder)
        # xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 -> B*N*C

        # # learnable null embedding
        # xyxy_null = self.null_position_feature.view(1, 1, -1)

        # # replace padding with learnable null embedding
        # xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        time_embeds = boxes

        # positionet with text only information
        if positive_embeddings is not None:
            # learnable null embedding
            positive_null = self.null_positive_feature.view(1, 1, -1)

            # replace padding with learnable null embedding
            positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null

            objs = self.linears(torch.cat([positive_embeddings, time_embeds], dim=-1))

        # positionet with text and image information
        else:
            raise NotImplementedError

        return objs
