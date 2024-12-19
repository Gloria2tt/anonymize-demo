import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import os.path as osp
import os
import numpy as np
from foleycrafter.models.auffusion_unet import UNet2DConditionModel
#from diffusers import UNet2DConditionModel
from ip_adpater_infer import IPAdapter
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer
import glob
from foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
from foleycrafter.utils.util import build_foleycrafter, read_frames_with_moviepy
from intern_preprocess import get_vid_detect,get_viclip,model_cfgs
from pathlib import Path
import soundfile as sf
from moviepy.editor import AudioFileClip, VideoFileClip
base_model_path = "auffusion/auffusion-full-no-adapter"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder/"
#ip_ckpt = 'cross_1108/checkpoint-200000/controlnet/ipa_combine.bin'
ip_ckpt = '/root/paddlejob/workspace/env_run/output/FoleyCrafter/cross_1108/checkpoint-100000/controlnet/ipa_combine.bin'
model_path = '/root/paddlejob/workspace/env_run/output/FoleyCrafter/att3_1108/checkpoint-50000/controlnet'
device = "cuda"

noise_scheduler = DDIMScheduler.from_pretrained(base_model_path,subfolder='scheduler')
vae = AutoencoderKL.from_pretrained(base_model_path,subfolder='vae')
unet = UNet2DConditionModel.from_pretrained(base_model_path,subfolder='unet',low_cpu_mem_usage=False,device_map=None)
# load SD pipeline 
vocoder = Generator.from_pretrained("./checkpoints", subfolder="vocoder").to(device).to(torch.float16)
text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder"
        ).to("cuda")

pipe = StableDiffusionPipeline(
    unet=unet,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
)    

pipe.to(torch.float16)                                               
save_dir = "./output_vggsound"
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, model_path, device)
video_path_list = glob.glob(f"examples/vggsound/*.mp4")
cfg = model_cfgs['viclip-l-internvid-10m-flt']
model_l = get_viclip(cfg['size'], cfg['pretrained'])
clip = model_l['viclip'].cuda()
#print("done")
image_processor = CLIPImageProcessor()
with torch.no_grad():
    for input_video in video_path_list:
        name = Path(input_video).stem
        name = name.replace("+", " ")
        internvideo_feat = get_vid_detect(input_video,clip,8).to(device)
        frames, duration = read_frames_with_moviepy(input_video, max_frame_nums=250)
        images = image_processor(images=frames,return_tensors="pt").to(device)
        image_embeddings = image_encoder(**images).image_embeds.to(device).unsqueeze(0)
        #image_embeddings_mean = torch.mean(image_embeddings, dim=0, keepdim=True).unsqueeze(0).to(dtype=pipe.unet.dtype)
        #print(image_embeddings_mean.shape) ### 1 x 1 x 1024
        #neg_image_embeddings = torch.zeros_like(image_embeddings)
        #image_embeddings = torch.cat([neg_image_embeddings, image_embeddings], dim=1).squeeze(0)
        #image_embeddings = image_embeddings.unsqueeze(0)
        neg_image_embeddings = torch.zeros_like(image_embeddings).to(device)
        encoder_hidden_states_2 = torch.cat([neg_image_embeddings,image_embeddings]).to(device) ### 2 x 250 x 1024
        encoder_hidden_states_2 = ip_model.video_proj_model(encoder_hidden_states_2).to(dtype=pipe.unet.dtype)
        print(encoder_hidden_states_2.shape) ### 2 x 250 x 768
        #print(image_embeddings.shape)
        #print("ok")
        #print("-----",encoder_hidden_states_2.shape) ### 2 x 1024
        #internvideo_feat = internvideo_feat.to(dtype=pipe.unet.dtype)
        images = ip_model.generate(clip_image_embeds=internvideo_feat,
                                    encoder_hidden_states_2=encoder_hidden_states_2,
                                    num_samples=1, 
                                    num_inference_steps=20, 
                                    seed=945, 
                                    strength=0.0,
                                    height=256,
                                    width=1024,output_type='pt'
                           )


        audio_img = images[0]
        audio = denormalize_spectrogram(audio_img).to(device)
        audio = vocoder.inference(audio, lengths=160000)[0]
        audio_save_path = osp.join(save_dir, "audio")
        video_save_path = osp.join(save_dir, "video")
        os.makedirs(audio_save_path, exist_ok=True)
        os.makedirs(video_save_path, exist_ok=True)
        audio = audio[: int(duration * 16000)]
        save_path = osp.join(audio_save_path, f"{name}.wav")
        sf.write(save_path, audio, 16000)
        audio = AudioFileClip(osp.join(audio_save_path, f"{name}.wav"))
        video = VideoFileClip(input_video)
        audio = audio.subclip(0, duration)
        video.audio = audio
        video = video.subclip(0, duration)
        os.makedirs(video_save_path, exist_ok=True)
        video.write_videofile(osp.join(video_save_path, f"{name}.mp4"))
        

