
import json
import numpy as np
import os
import cv2

import torch
from viclip import get_viclip, retrieve_text, _frame_from_video


model_cfgs = {
    'viclip-l-internvid-10m-flt': {
        'size': 'l',
        'pretrained': '../ViCLIP/ViCLIP-L_InternVid-FLT-10M.pth',
    },
    'viclip-l-internvid-200m': {
        'size': 'l',
        'pretrained': 'xxx/ViCLIP-L_InternVid-200M.pth',
    },
    'viclip-b-internvid-10m-flt': {
        'size': 'b',
        'pretrained': 'xxx/ViCLIP-B_InternVid-FLT-10M.pth',
    },
    'viclip-b-internvid-200m': {
        'size': 'b',
        'pretrained': 'xxx/ViCLIP-B_InternVid-200M.pth',
    },
}
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)

def get_vid_detect(mp4_path,clip,target_num):
    video = cv2.VideoCapture(mp4_path)
    original_fps = video.get(cv2.CAP_PROP_FPS)
    #frame_count = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #frame_interval = int(original_fps / target_fps)###15
    #print(frame_interval,total_frames)
    frames = [x for x in _frame_from_video(video)]
    sample_step = len(frames) // target_num
    sampled_frames = frames[::sample_step][:target_num]
    frames_tensor = frames2tensor(frames)
    with torch.no_grad():
        vid_feat = get_vid_feat(frames_tensor, clip)

    return vid_feat


if __name__=="__main__":
    cfg = model_cfgs['viclip-l-internvid-10m-flt']
    model_l = get_viclip(cfg['size'], cfg['pretrained'])
    clip = model_l['viclip'].to('cuda')
    video_feat = get_vid_detect('/root/paddlejob/workspace/env_run/output/FoleyCrafter/examples/avsync/0.mp4',clip,20)
    print(video_feat.shape)


