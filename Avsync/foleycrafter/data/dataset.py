import glob
import io
import pickle
import random
import os 
import numpy as np
import torch
import torch.distributed as dist
import torchaudio
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torchaudio

def zero_rank_print(s):
    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
        print("### " + s, flush=True)


@torch.no_grad()
def get_mel(audio_data, audio_cfg):
    # mel shape: (n_mels, T)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg["sample_rate"],
        n_fft=audio_cfg["window_size"],
        win_length=audio_cfg["window_size"],
        hop_length=audio_cfg["hop_size"],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=64,
        f_min=audio_cfg["fmin"],
        f_max=audio_cfg["fmax"],
    ).to(audio_data.device)
    mel = mel(audio_data)
    # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel  # (T, n_mels)


def dynamic_range_compression(x, normalize_fun=torch.log, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return normalize_fun(torch.clamp(x, min=clip_val) * C)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class AudioSetStrong(Dataset):
    # read feature and audio
    def __init__(
        self,
        data_path="data/AudioSetStrong/train/feature",
        video_path="data/AudioSetStrong/train/video",
    ):
        super().__init__()
        self.data_path = data_path
        self.data_list = list(self.data_path)
        self.length = len(self.data_list)
        # get video feature
        self.video_path = video_path
        vision_transform_list = [
            transforms.Resize((128, 128)),
            transforms.CenterCrop((112, 112)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.video_transform = transforms.Compose(vision_transform_list)

    def get_batch(self, idx):
        embeds = self.data_list[idx]
        mel = embeds["mel"]
        save_bsz = mel.shape[0]
        audio_info = embeds["audio_info"]
        text_embeds = embeds["text_embeds"]

        # audio_info['label_list'] = np.array(audio_info['label_list'])
        audio_info_array = np.array(audio_info["label_list"])
        prompts = []
        for i in range(save_bsz):
            prompts.append(", ".join(audio_info_array[i, : audio_info["event_num"][i]].tolist()))

        return mel, audio_info, text_embeds, prompts

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                mel, audio_info, text_embeds, prompts, videos = self.get_batch(idx)
                break
            except Exception:
                zero_rank_print(" >>> load error <<<")
                idx = random.randint(0, self.length - 1)
        sample = {
            "mel": mel,
            "audio_info": audio_info,
            "text_embeds": text_embeds,
            "prompts": prompts,
            "videos": videos,
        }
        return sample


class VGGSound(Dataset):
    # read feature and audio
    def __init__(
        self,
        data_path="data/VGGSound/train/video",
        visual_data_path="data/VGGSound/train/feature",
    ):
        super().__init__()
        self.data_path = data_path
        self.visual_data_path = visual_data_path
        self.embeds_list = glob.glob(f"{self.data_path}/*.pt")
        self.visual_list = glob.glob(f"{self.visual_data_path}/*.pt")
        self.length = len(self.embeds_list)

    def get_batch(self, idx):
        embeds = torch.load(self.embeds_list[idx], map_location="cpu")
        visual_embeds = torch.load(self.visual_list[idx], map_location="cpu")
        
        # audio_embeds  = embeds['audio_embeds']
        visual_embeds = visual_embeds["visual_embeds"]
        # video_name = embeds["video_name"]
        text = embeds["text"]
        mel = embeds["mel"]

        audio = mel

        return visual_embeds, audio, text

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                visual_embeds, audio, text = self.get_batch(idx)
                break
            except Exception:
                zero_rank_print("load error")
                idx = random.randint(0, self.length - 1)
        sample = {"visual_embeds": visual_embeds, "audio": audio, "text": text}
        return sample


"""class Mydataset(Dataset):
    def __init__(self, dataset_path, min_duration=3, max_duration=5):
        super().__init__()
        self.path_split_name = os.listdir(dataset_path)
        self.dataset_path = dataset_path
        self.audio_root = "../../../audio"
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # 创建mel spectrogram转换器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,  # 根据你的音频采样率调整
            n_fft=2048,
            hop_length=512,
            n_mels=256,
            power=1,
            normalized=False
        )
    
    def __len__(self):
        return len(self.path_split_name)
        
    def get_batch(self, idx):
        path_name = self.path_split_name[idx]
        n = path_name.split(".")[0]
        npy_path = os.path.join(self.dataset_path, path_name)
        audio_path = os.path.join(self.audio_root, n+'.wav')
        
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True) 
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        # 随机选择音频片段
        prompt_duration = random.uniform(self.min_duration, self.max_duration)
        prompt_frames = int(prompt_duration * 16000)
        total_frames = waveform.shape[1]
        prompt_frames = int(random.uniform(self.min_duration, self.max_duration) * 16000)

        if total_frames <= prompt_frames:
            prompt_waveform = waveform
        else:
            max_start = total_frames - prompt_frames
            start_frame = random.randint(0, max_start)
            prompt_waveform = waveform[:1, start_frame:start_frame + prompt_frames]

        # 提取并处理mel spectrogram
        prompt_mel = self.mel_transform(prompt_waveform)
        prompt_mel = (prompt_mel - prompt_mel.mean()) / (prompt_mel.std() + 1e-8)

        # 调整大小并扩展通道
        if prompt_mel.size(2) < 1024:
            prompt_mel = torch.nn.functional.pad(prompt_mel, (0, 1024 - prompt_mel.size(2)))
        else:
            prompt_mel = prompt_mel[:, :, :1024]
        
        
        # 读取其他数据
        data = np.load(npy_path)
        visual_embeds = torch.from_numpy(data['visual_embeds'])
        visual_embeds = visual_embeds.mean(dim=0)
        mel = torch.from_numpy(data['mel'])
        text = data['text']
        audio_embeds = torch.from_numpy(data['audio_embeds'])
        
        return visual_embeds, text, audio_embeds, mel, prompt_mel
    
    def __getitem__(self, idx):
        visual_embeds, text, audio_embeds, mel, prompt_mel = self.get_batch(idx)
        
        sample = {
            "visual_embeds": visual_embeds, 
            "mel": mel, 
            "text": text,
            'audio_embeds': audio_embeds,
            'prompt_mel': prompt_mel
        }
        
        return sample

    def collate_fn(batch):
        visual_embeds = torch.stack([item['visual_embeds'] for item in batch])
        mel = torch.stack([item['mel'] for item in batch])
        prompt_mel = torch.stack([item['prompt_mel'] for item in batch])
        audio_embeds = torch.stack([item['audio_embeds'] for item in batch])
        
        texts = [str(item['text']) for item in batch]
        
        return {
            'visual_embeds': visual_embeds,
            'mel': mel,
            'text': texts,
            'audio_embeds': audio_embeds,
            'prompt_mel': prompt_mel
        }"""

from encodec import EncodecModel
from encodec.utils import convert_audio
class Mydataset(Dataset):
    def __init__(self, dataset_path, min_duration=3, max_duration=5):
        super().__init__()
        self.path_split_name = os.listdir(dataset_path)
        self.dataset_path = dataset_path
        self.audio_root = "/root/paddlejob/workspace/env_run/output/encodec_dataset_all"
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.cfg_training = True
        #self.encodec = EncodecModel.encodec_model_24khz()
        
    
    def __len__(self):
        return len(self.path_split_name)

    def pad_video(self, video, target_length):
        """填充视频帧至目标长度"""
        current_length = video.shape[0]
        if current_length >= target_length:
            return video
        
        # 计算需要填充的帧数
        pad_length = target_length - current_length
        
        # 创建填充张量 (使用最后一帧重复填充)
        pad_frames = video[-1].unsqueeze(0).repeat(pad_length, 1)
        
        # 连接原始视频和填充帧
        padded_video = torch.cat([video, pad_frames], dim=0)
        return padded_video
    
    def process_video(self, video, target_frames=250):
        """处理视频使其符合目标帧数"""
        if video.shape[0] > target_frames:
            # 如果帧数过多，进行均匀采样
            indices = np.linspace(0, video.shape[0]-1, target_frames, dtype=int)
            video = video[indices]
        elif video.shape[0] < target_frames:
            # 如果帧数不足，进行填充
            video = self.pad_video(video, target_frames)
        return video
        
    def get_batch(self, idx):
        path_name = self.path_split_name[idx]
        npy_path = os.path.join(self.dataset_path, path_name)
        audio_en_path = os.path.join(self.audio_root,path_name)
        data = np.load(npy_path)
        data_encodec = np.load(audio_en_path)['embedding']
        visual_embeds = torch.from_numpy(data['internvl'])
        encodec_emb = torch.from_numpy(data_encodec)
        #visual_embeds = visual_embeds.mean(dim=0) #### 1024
        mel = torch.from_numpy(data['mel'])
        text = data['text']
        video = torch.from_numpy(data['visual_embeds'])
        #print(video.shape)
        video = self.process_video(video)
        #audio_path = os.path.join(self.audio_root,name_)
        audio_emb = torch.from_numpy(data['audio_embeds']).unsqueeze(0)

        
        
        return visual_embeds, text, video, mel, audio_emb,encodec_emb
    
    def __getitem__(self, idx):
        visual_embeds, text, video_embeds, mel, audio_emb, encodec_emb = self.get_batch(idx)
        if self.cfg_training and np.random.random() < 0.05:  # 10%的概率
            empty_visual = torch.zeros_like(visual_embeds)
            empty_video = torch.zeros_like(video_embeds)
            empty_text = ""
            empty_audio = torch.zeros_like(audio_emb)
            encodec_emb = torch.zeros_like(encodec_emb)
            
            sample = {
                "visual_embeds": empty_visual,
                "mel": mel,
                "text": empty_text,
                'video_embeds': empty_video,
                'audio_embeds': empty_audio,
                'encodec_embeds': encodec_emb
            }
        elif 0.1< self.cfg_training and np.random.random() < 0.2:
            sample = {
                "visual_embeds": visual_embeds,
                "mel": mel,
                "text": '',
                'video_embeds': video_embeds,
                'audio_embeds': audio_emb,
                'encodec_embeds': encodec_emb
            }
        else:
            sample = {
                "visual_embeds": visual_embeds,
                "mel": mel,
                "text": text,
                'video_embeds': video_embeds,
                'audio_embeds': audio_emb,
                'encodec_embeds': encodec_emb
            }
        
        return sample

    def collate_fn(batch):
        """
        自定义batch处理函数
        """
        visual_embeds = torch.stack([item['visual_embeds'] for item in batch])
        mel = torch.stack([item['mel'] for item in batch])
        video_embeds = torch.stack([item['video_embeds'] for item in batch])
        texts = [str(item['text']) for item in batch]
        audio_embeds = torch.stack([item['audio_embeds'] for item in batch])
        encodec_emb = torch.stack([item['encodec_embeds'] for item in batch])
        return {
            'visual_embeds': visual_embeds,
            'mel': mel,
            'text': texts,
            'video_embeds': video_embeds,
            'audio_embeds': audio_embeds,
            'encodec_embeds': encodec_emb
        }
# 使用示例
if __name__ == "__main__":
    # 创建数据集实例
    dataset = Mydataset(dataset_path="../../../processed_dataset")
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=Mydataset.collate_fn
    )
    
    # 测试一个batch
    for batch in dataloader:
        print("Visual embeds shape:", batch['visual_embeds'].shape)
        print("Mel spectrogram shape:", batch['mel'].shape)
        print("Audio embeds shape:", batch['video_embeds'].shape)
        print("Text example:", batch['text'][0])
        print("Audio embeds shape:", batch['audio_embeds'].shape)
        print(batch['encodec_embeds'].shape)
        break
                


