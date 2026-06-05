"""
Civis Lucri-Faber 听觉/发音系统训练脚本
=========================================

使用 RAVDESS + MINDS-14 + 4-lang emotional speech 数据集

Usage:
    python train_audio.py --mode auditory --epochs 50
    python train_audio.py --mode phonetic --epochs 30
    python train_audio.py --mode vocal --epochs 30
    python train_audio.py --mode binaural --epochs 30
    python train_audio.py --mode language --epochs 30
    python train_audio.py --mode full --epochs 50
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.auditory_cortex import AuditoryCortex
from core.phonetic_perception import PhoneticPerception
from core.vocalization import VocalCortex
from core.spiking_auditory_cortex import SpikingAuditoryCortex
from core.cognitive_auditory_cortex import CognitiveAuditoryCortex
from core.binaural_auditory import BinauralProcessor
from core.language_cortex import LanguageCortex

# ─── 常量 ────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
AUDIO_LEN = 32000  # 2秒

RAVDESS_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise']
EMOTION_VAD = {
    'neutral': (0.0, 0.3), 'calm': (0.4, 0.2), 'happy': (0.8, 0.7),
    'sad': (-0.7, 0.3), 'angry': (-0.6, 0.9), 'fearful': (-0.5, 0.85),
    'disgust': (-0.6, 0.5), 'surprise': (0.2, 0.8),
}

EMOTION_4LANG_MAP = {
    'angry': 'angry', 'disgust': 'disgust', 'fear': 'fearful',
    'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad',
    'surprise': 'surprise', 'calm': 'calm',
}


# ─── 音频工具 ─────────────────────────────────────────────────────
def resample_audio(waveform, orig_sr, target_sr):
    if orig_sr == target_sr:
        return waveform
    duration = len(waveform) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(waveform) - 1, target_len)
    return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)


def pad_crop_audio(waveform, length, train=True):
    if len(waveform) >= length:
        if train:
            start = np.random.randint(0, len(waveform) - length + 1)
            return waveform[start:start + length]
        else:
            start = (len(waveform) - length) // 2
            return waveform[start:start + length]
    else:
        padded = np.zeros(length, dtype=np.float32)
        padded[:len(waveform)] = waveform
        return padded


def load_audio_hf(audio_dict, target_sr=16000, length=32000, train=True):
    arr = np.array(audio_dict['array'], dtype=np.float32)
    sr = audio_dict.get('sampling_rate', 16000)
    arr = resample_audio(arr, sr, target_sr)
    peak = np.max(np.abs(arr)) + 1e-8
    arr = arr / peak
    return pad_crop_audio(arr, length, train)


def load_audio_wf(wf_data, target_sr=16000, length=32000, train=True):
    """从RAVDESS wf列加载音频 (可能是bytes dict或numpy array)"""
    import soundfile as sf
    import io
    if isinstance(wf_data, dict) and 'bytes' in wf_data:
        arr, sr = sf.read(io.BytesIO(wf_data['bytes']))
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
    else:
        arr = np.array(wf_data, dtype=np.float32)
        sr = 16000
    arr = resample_audio(arr, sr, target_sr)
    peak = np.max(np.abs(arr)) + 1e-8
    arr = arr / peak
    return pad_crop_audio(arr, length, train)


def text_to_phoneme_indices(text, max_len=32):
    tokens = []
    for ch in text[:max_len]:
        tokens.append(ord(ch) % 40)
    while len(tokens) < max_len:
        tokens.append(0)
    return torch.LongTensor(tokens[:max_len])


def text_to_tokens(text, vocab_size=10000, max_len=64):
    words = text.lower().split()[:max_len]
    tokens = [hash(w) % vocab_size for w in words]
    while len(tokens) < max_len:
        tokens.append(0)
    return torch.LongTensor(tokens[:max_len])


# ─── 数据集类 ─────────────────────────────────────────────────────
class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, train=True):
        path = data_dir / 'ravdess'
        self.data = load_from_disk(str(path)) if path.exists() else None
        self.train = train

    def __len__(self):
        return len(self.data) if self.data else 0

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio = load_audio_wf(sample['wf'], train=self.train)
        label = sample['label']
        vad = EMOTION_VAD[RAVDESS_EMOTIONS[label]]
        return {
            'audio': torch.FloatTensor(audio),
            'emotion_label': torch.LongTensor([label]),
            'valence': torch.FloatTensor([vad[0]]),
            'arousal': torch.FloatTensor([vad[1]]),
        }


class Emotional4LangDataset(Dataset):
    def __init__(self, data_dir, train=True):
        path = data_dir / 'emotional_speech_4lang'
        if path.exists():
            ds = load_from_disk(str(path))
            table = ds._data
            self.audio_bytes = [table['audio'][i]['bytes'].as_py() for i in range(len(ds))]
            self.emotions = [table['emotion'][i].as_py() for i in range(len(ds))]
        else:
            self.audio_bytes = []
            self.emotions = []
        self.train = train

    def __len__(self):
        return len(self.audio_bytes)

    def __getitem__(self, idx):
        import soundfile as sf
        import io
        arr, sr = sf.read(io.BytesIO(self.audio_bytes[idx]))
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        arr = resample_audio(arr, sr, 16000)
        peak = np.max(np.abs(arr)) + 1e-8
        arr = arr / peak
        audio = pad_crop_audio(arr, AUDIO_LEN, self.train)

        emo_str = self.emotions[idx]
        mapped = EMOTION_4LANG_MAP.get(emo_str, 'neutral')
        label = RAVDESS_EMOTIONS.index(mapped) if mapped in RAVDESS_EMOTIONS else 0
        vad = EMOTION_VAD[mapped]
        return {
            'audio': torch.FloatTensor(audio),
            'emotion_label': torch.LongTensor([label]),
            'valence': torch.FloatTensor([vad[0]]),
            'arousal': torch.FloatTensor([vad[1]]),
        }


class MINDS14Dataset(Dataset):
    def __init__(self, data_dir, train=True):
        path = data_dir / 'minds14'
        if path.exists():
            ds = load_from_disk(str(path))
            table = ds._data
            self.audio_bytes = [table['audio'][i]['bytes'].as_py() for i in range(len(ds))]
            self.intents = [table['intent_class'][i].as_py() for i in range(len(ds))]
        else:
            self.audio_bytes = []
            self.intents = []
        self.train = train

    def __len__(self):
        return len(self.audio_bytes)

    def __getitem__(self, idx):
        import soundfile as sf
        import io
        arr, sr = sf.read(io.BytesIO(self.audio_bytes[idx]))
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        arr = resample_audio(arr, sr, 16000)
        peak = np.max(np.abs(arr)) + 1e-8
        arr = arr / peak
        audio = pad_crop_audio(arr, AUDIO_LEN, self.train)
        return {
            'audio': torch.FloatTensor(audio),
            'intent_label': torch.LongTensor([self.intents[idx]]),
        }


class TextAudioDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.train = train
        minds_path = data_dir / 'minds14'
        if minds_path.exists():
            ds = load_from_disk(str(minds_path))
            table = ds._data
            self.audio_bytes = [table['audio'][i]['bytes'].as_py() for i in range(len(ds))]
            self.texts = [table['english_transcription'][i].as_py() if 'english_transcription' in table.schema.names else '' for i in range(len(ds))]
        else:
            self.audio_bytes = []
            self.texts = []

    def __len__(self):
        return len(self.audio_bytes)

    def __getitem__(self, idx):
        import soundfile as sf
        import io
        arr, sr = sf.read(io.BytesIO(self.audio_bytes[idx]))
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        arr = resample_audio(arr, sr, 16000)
        peak = np.max(np.abs(arr)) + 1e-8
        arr = arr / peak
        audio = pad_crop_audio(arr, AUDIO_LEN, self.train)
        phonemes = text_to_phoneme_indices(self.texts[idx])
        tokens = text_to_tokens(self.texts[idx])
        return {
            'audio': torch.FloatTensor(audio),
            'phonemes': phonemes,
            'tokens': tokens,
        }


class SyntheticBinauralDataset(Dataset):
    def __init__(self, data_dir, n_channels=128, train=True):
        self.mono_audios = []
        self.train = train
        self.n_channels = n_channels
        ravdess_path = data_dir / 'ravdess'
        if ravdess_path.exists():
            data = load_from_disk(str(ravdess_path)).with_format(None)
            for s in data:
                wf = s['wf']
                if isinstance(wf, dict) and 'bytes' in wf:
                    import soundfile as sf
                    import io
                    arr, _ = sf.read(io.BytesIO(wf['bytes']))
                    arr = arr.astype(np.float32)
                    if arr.ndim > 1:
                        arr = arr.mean(axis=1)
                else:
                    arr = np.array(wf, dtype=np.float32)
                self.mono_audios.append(arr)

    def __len__(self):
        return len(self.mono_audios)

    def __getitem__(self, idx):
        mono = self.mono_audios[idx]
        peak = np.max(np.abs(mono)) + 1e-8
        mono = mono / peak
        mono = pad_crop_audio(mono, AUDIO_LEN, self.train)

        azimuth = np.random.uniform(-90, 90)
        max_delay = 11
        delay_samples = int(abs(azimuth) / 90 * max_delay)
        ild_db = abs(azimuth) / 90 * 10
        ild_factor = 10 ** (-ild_db / 20)

        if azimuth >= 0:
            right = mono
            left = np.roll(mono, delay_samples) * ild_factor
        else:
            left = mono
            right = np.roll(mono, delay_samples) * ild_factor

        # 简单频带分解 [T, C]
        n_fft = 256
        hop = 128
        n_time = AUDIO_LEN // hop
        left_feat = np.zeros((n_time, self.n_channels), dtype=np.float32)
        right_feat = np.zeros((n_time, self.n_channels), dtype=np.float32)
        for t in range(n_time):
            chunk_l = left[t * hop:(t + 1) * hop] * np.hanning(hop)
            chunk_r = right[t * hop:(t + 1) * hop] * np.hanning(hop)
            spec_l = np.abs(np.fft.rfft(chunk_l, n=n_fft))
            spec_r = np.abs(np.fft.rfft(chunk_r, n=n_fft))
            # 映射到n_channels
            c_len = min(len(spec_l), self.n_channels)
            left_feat[t, :c_len] = spec_l[:c_len]
            right_feat[t, :c_len] = spec_r[:c_len]

        return {
            'left': torch.FloatTensor(left_feat),
            'right': torch.FloatTensor(right_feat),
            'azimuth': torch.FloatTensor([azimuth]),
        }


# ─── 训练器 ──────────────────────────────────────────────────────
class AudioTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] Device: {self.device}")
        if torch.cuda.is_available():
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = args.mode

        self._load_datasets()
        self._init_modules()

        self.history = {
            'loss': [], 'emotion_acc': [], 'intent_acc': [],
            'azimuth_mae': [], 'gpu_memory': [], 'epoch_time': [],
        }

    def _load_datasets(self):
        data_dir = PROJECT_ROOT / self.args.data_dir
        self.ravdess_ds = RAVDESSDataset(data_dir)
        self.emotional4_ds = Emotional4LangDataset(data_dir)
        self.minds14_ds = MINDS14Dataset(data_dir)
        self.text_ds = TextAudioDataset(data_dir)
        self.binaural_ds = SyntheticBinauralDataset(data_dir)

        print(f"[Data] RAVDESS: {len(self.ravdess_ds)} | 4-lang: {len(self.emotional4_ds)} | MINDS-14: {len(self.minds14_ds)} | Text: {len(self.text_ds)} | Binaural: {len(self.binaural_ds)}")

    def _init_modules(self):
        lr = self.args.lr
        self.modules = {}
        self.optimizers = {}
        self.probes = {}

        if self.mode in ['auditory', 'full']:
            self.modules['auditory'] = AuditoryCortex().to(self.device)
            self.modules['spiking'] = SpikingAuditoryCortex().to(self.device)
            self.modules['cognitive'] = CognitiveAuditoryCortex().to(self.device)
            self.probes['emotion'] = nn.Linear(4, 8).to(self.device)
            self.probes['intent'] = nn.Linear(64, 14).to(self.device)
            self.probes['spike_recon'] = nn.Linear(256, 256).to(self.device)
            all_params = (
                list(self.modules['auditory'].parameters()) +
                list(self.modules['spiking'].parameters()) +
                list(self.modules['cognitive'].parameters()) +
                list(self.probes['emotion'].parameters()) +
                list(self.probes['intent'].parameters()) +
                list(self.probes['spike_recon'].parameters())
            )
            self.optimizers['auditory'] = optim.Adam(all_params, lr=lr)

        if self.mode in ['phonetic', 'full']:
            self.modules['phonetic'] = PhoneticPerception(input_dim=256).to(self.device)
            self.optimizers['phonetic'] = optim.Adam(self.modules['phonetic'].parameters(), lr=lr)

        if self.mode in ['vocal', 'full']:
            self.modules['vocal'] = VocalCortex().to(self.device)
            self.optimizers['vocal'] = optim.Adam(self.modules['vocal'].parameters(), lr=lr)

        if self.mode in ['binaural', 'full']:
            self.modules['binaural'] = BinauralProcessor().to(self.device)
            self.optimizers['binaural'] = optim.Adam(self.modules['binaural'].parameters(), lr=lr)

        if self.mode in ['language', 'full']:
            self.modules['language'] = LanguageCortex(use_parallel=True).to(self.device)
            self.optimizers['language'] = optim.Adam(self.modules['language'].parameters(), lr=lr)

        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ─── 训练epoch ────────────────────────────────────────────────

    def train_auditory_epoch(self, epoch):
        losses = {}
        if len(self.ravdess_ds) == 0:
            return losses

        emotion_data = list(self.ravdess_ds) + list(self.emotional4_ds)
        emotion_loader = DataLoader(emotion_data, batch_size=self.args.batch_size, shuffle=True)
        intent_loader = DataLoader(list(self.minds14_ds), batch_size=self.args.batch_size, shuffle=True)

        total_emo_loss = 0.0
        emo_correct = 0
        emo_total = 0
        total_distill = 0.0
        steps = 0

        # 情绪分类
        self.modules['auditory'].train()
        self.modules['spiking'].train()
        for batch in emotion_loader:
            audio = batch['audio'].to(self.device)
            labels = batch['emotion_label'].squeeze(-1).to(self.device)
            valence = batch['valence'].squeeze(-1).to(self.device)
            arousal = batch['arousal'].squeeze(-1).to(self.device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = self.modules['auditory'](audio)
                vad = torch.stack([out['valence'], out['arousal'],
                                  out['dominance'], out['pleasantness']], dim=-1)
                emo_pred = self.probes['emotion'](vad)
                emo_loss = F.cross_entropy(emo_pred, labels)
                vad_loss = F.mse_loss(out['valence'], valence) + F.mse_loss(out['arousal'], arousal)

                spike_out = self.modules['spiking'](audio)
                spike_feat = self.probes['spike_recon'](spike_out['features'])
                distill_loss = F.mse_loss(spike_feat, out['features'].detach())

                loss = emo_loss + 0.3 * vad_loss + 0.1 * distill_loss

            self.optimizers['auditory'].zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizers['auditory'])
            torch.nn.utils.clip_grad_norm_(self.optimizers['auditory'].param_groups[0]['params'], 1.0)
            self.scaler.step(self.optimizers['auditory'])
            self.scaler.update()

            total_emo_loss += emo_loss.item()
            total_distill += distill_loss.item()
            emo_correct += (emo_pred.argmax(-1) == labels).sum().item()
            emo_total += len(labels)
            steps += 1

        losses['emotion_loss'] = total_emo_loss / max(steps, 1)
        losses['emotion_acc'] = emo_correct / max(emo_total, 1)
        losses['distill_loss'] = total_distill / max(steps, 1)

        # 意图分类
        if len(self.minds14_ds) > 0:
            self.modules['cognitive'].train()
            int_loss_total = 0.0
            int_correct = 0
            int_total = 0
            int_steps = 0

            for batch in intent_loader:
                audio = batch['audio'].to(self.device)
                intents = batch['intent_label'].squeeze(-1).to(self.device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out = self.modules['cognitive'](audio)
                    intent_pred = self.probes['intent'](out['output'])
                    intent_loss = F.cross_entropy(intent_pred, intents)

                self.optimizers['auditory'].zero_grad()
                self.scaler.scale(intent_loss).backward()
                self.scaler.unscale_(self.optimizers['auditory'])
                torch.nn.utils.clip_grad_norm_(self.optimizers['auditory'].param_groups[0]['params'], 1.0)
                self.scaler.step(self.optimizers['auditory'])
                self.scaler.update()

                int_loss_total += intent_loss.item()
                int_correct += (intent_pred.argmax(-1) == intents).sum().item()
                int_total += len(intents)
                int_steps += 1

            losses['intent_loss'] = int_loss_total / max(int_steps, 1)
            losses['intent_acc'] = int_correct / max(int_total, 1)

        return losses

    def train_phonetic_epoch(self, epoch):
        if len(self.text_ds) == 0:
            return {}

        if 'auditory' not in self.modules:
            self.modules['auditory'] = AuditoryCortex().to(self.device)
            self.modules['auditory'].eval()
            for p in self.modules['auditory'].parameters():
                p.requires_grad = False

        loader = DataLoader(list(self.text_ds), batch_size=self.args.batch_size, shuffle=True)
        total_loss = 0.0
        steps = 0

        self.modules['phonetic'].train()
        for batch in loader:
            audio = batch['audio'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)

            with torch.no_grad():
                aud_out = self.modules['auditory'](audio)
                features = aud_out['features']

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pho_out = self.modules['phonetic'](features)
                target = phonemes[:, 0]
                loss = F.cross_entropy(pho_out['phoneme'].unsqueeze(0), target.unsqueeze(0))

            self.optimizers['phonetic'].zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizers['phonetic'])
            torch.nn.utils.clip_grad_norm_(self.modules['phonetic'].parameters(), 1.0)
            self.scaler.step(self.optimizers['phonetic'])
            self.scaler.update()

            total_loss += loss.item()
            steps += 1

        return {'phonetic_loss': total_loss / max(steps, 1)}

    def train_vocal_epoch(self, epoch):
        if len(self.text_ds) == 0:
            return {}

        loader = DataLoader(list(self.text_ds), batch_size=self.args.batch_size, shuffle=True)
        total_loss = 0.0
        steps = 0

        self.modules['vocal'].train()
        for batch in loader:
            audio = batch['audio'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = self.modules['vocal'](phoneme_indices=phonemes)
                if 'auditory' not in self.modules:
                    self.modules['auditory'] = AuditoryCortex().to(self.device)
                with torch.no_grad():
                    aud_out = self.modules['auditory'](audio)
                    target_feat = aud_out['features']

                if out.get('acoustic_features') is not None:
                    pred = out['acoustic_features']
                    if pred.shape[-1] != target_feat.shape[-1]:
                        pred = pred.mean(dim=-1)
                        target_feat = target_feat.mean(dim=-1)
                    loss = F.mse_loss(pred, target_feat)
                else:
                    loss = torch.tensor(0.01, device=self.device)

            self.optimizers['vocal'].zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizers['vocal'])
            torch.nn.utils.clip_grad_norm_(self.modules['vocal'].parameters(), 1.0)
            self.scaler.step(self.optimizers['vocal'])
            self.scaler.update()

            total_loss += loss.item()
            steps += 1

        return {'vocal_loss': total_loss / max(steps, 1)}

    def train_binaural_epoch(self, epoch):
        if len(self.binaural_ds) == 0:
            return {}

        loader = DataLoader(list(self.binaural_ds), batch_size=self.args.batch_size, shuffle=True)
        total_loss = 0.0
        total_mae = 0.0
        steps = 0

        self.modules['binaural'].train()
        for batch in loader:
            left = batch['left'].to(self.device)
            right = batch['right'].to(self.device)
            target_az = batch['azimuth'].squeeze(-1).to(self.device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = self.modules['binaural'](left, right)
                pred_az = out['azimuth']
                loss = F.mse_loss(pred_az, target_az)

            self.optimizers['binaural'].zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizers['binaural'])
            torch.nn.utils.clip_grad_norm_(self.modules['binaural'].parameters(), 1.0)
            self.scaler.step(self.optimizers['binaural'])
            self.scaler.update()

            total_loss += loss.item()
            total_mae += F.l1_loss(pred_az, target_az).item()
            steps += 1

        return {
            'binaural_loss': total_loss / max(steps, 1),
            'azimuth_mae': total_mae / max(steps, 1),
        }

    def train_language_epoch(self, epoch):
        if len(self.text_ds) == 0:
            return {}

        loader = DataLoader(list(self.text_ds), batch_size=self.args.batch_size, shuffle=True)
        total_loss = 0.0
        steps = 0

        self.modules['language'].train()
        for batch in loader:
            tokens = batch['tokens'].to(self.device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = self.modules['language'](tokens)
                features = out['features']

                if features.shape[0] > 1:
                    target = features[1:].detach()
                    pred = features[:-1]
                    loss = F.mse_loss(pred, target)
                else:
                    loss = features.pow(2).mean() * 0.01

            self.optimizers['language'].zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizers['language'])
            torch.nn.utils.clip_grad_norm_(self.modules['language'].parameters(), 1.0)
            self.scaler.step(self.optimizers['language'])
            self.scaler.update()

            total_loss += loss.item()
            steps += 1

        return {'language_loss': total_loss / max(steps, 1)}

    # ─── 主循环 ────────────────────────────────────────────────────

    def train_epoch(self, epoch):
        epoch_start = time.time()
        losses = {}

        if self.mode in ['auditory', 'full']:
            losses.update(self.train_auditory_epoch(epoch))
        if self.mode in ['phonetic', 'full']:
            losses.update(self.train_phonetic_epoch(epoch))
        if self.mode in ['vocal', 'full']:
            losses.update(self.train_vocal_epoch(epoch))
        if self.mode in ['binaural', 'full']:
            losses.update(self.train_binaural_epoch(epoch))
        if self.mode in ['language', 'full']:
            losses.update(self.train_language_epoch(epoch))

        if torch.cuda.is_available():
            self.history['gpu_memory'].append(torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.reset_peak_memory_stats()

        self.history['epoch_time'].append(time.time() - epoch_start)
        for k in ['emotion_acc', 'intent_acc', 'azimuth_mae']:
            if k in losses:
                self.history[k].append(losses[k])

        return losses

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {'epoch': epoch, 'args': vars(self.args), 'history': self.history}
        for name, mod in self.modules.items():
            checkpoint[f'{name}_state'] = mod.state_dict()
        for name, probe in self.probes.items():
            checkpoint[f'probe_{name}_state'] = probe.state_dict()
        for name, opt in self.optimizers.items():
            checkpoint[f'opt_{name}_state'] = opt.state_dict()
        if self.scaler:
            checkpoint['scaler_state'] = self.scaler.state_dict()

        torch.save(checkpoint, self.output_dir / 'latest.pt')
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')
        if epoch % 10 == 0:
            torch.save(checkpoint, self.output_dir / f'model_epoch_{epoch}.pt')

    def load_checkpoint(self, path):
        if not path.exists():
            print(f"[Warning] Checkpoint not found: {path}")
            return 0
        ckpt = torch.load(path, map_location=self.device)
        for name, mod in self.modules.items():
            if f'{name}_state' in ckpt:
                mod.load_state_dict(ckpt[f'{name}_state'])
        for name, probe in self.probes.items():
            if f'probe_{name}_state' in ckpt:
                probe.load_state_dict(ckpt[f'probe_{name}_state'])
        for name, opt in self.optimizers.items():
            if f'opt_{name}_state' in ckpt:
                opt.load_state_dict(ckpt[f'opt_{name}_state'])
        if 'scaler_state' in ckpt and self.scaler:
            self.scaler.load_state_dict(ckpt['scaler_state'])
        self.history = ckpt.get('history', self.history)
        print(f"[Checkpoint] Loaded from epoch {ckpt['epoch']}")
        return ckpt['epoch']

    def save_history(self):
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Civis Audio Training')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['auditory', 'phonetic', 'vocal', 'binaural', 'language', 'full'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/audio')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("Civis Lucri-Faber Audio Training")
    print("=" * 60)
    print(f"Mode: {args.mode} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")

    trainer = AudioTrainer(args)

    start_epoch = 1
    if args.resume:
        loaded = trainer.load_checkpoint(Path(args.resume))
        if loaded > 0:
            start_epoch = loaded + 1

    best_metric = 0.0
    total_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 60}\nEpoch {epoch}/{args.epochs}\n{'=' * 60}")

        losses = trainer.train_epoch(epoch)
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        print(f"  [Losses] {loss_str}")

        if epoch % args.log_interval == 0:
            metric = losses.get('emotion_acc', losses.get('intent_acc', losses.get('azimuth_mae', 0)))
            is_best = metric > best_metric
            if is_best:
                best_metric = metric
            trainer.save_checkpoint(epoch, is_best=is_best)

        if torch.cuda.is_available():
            print(f"  [GPU] {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        print(f"  [Time] {trainer.history['epoch_time'][-1]:.2f}s")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}\nTraining Completed!\n{'=' * 60}")
    print(f"Total: {total_time:.1f}s ({total_time/60:.1f} min) | Best: {best_metric:.4f}")

    trainer.save_history()
    trainer.save_checkpoint(args.epochs)


if __name__ == '__main__':
    main()