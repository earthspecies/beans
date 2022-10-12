import json
import pandas as pd

from memoization import cached
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from beans.torchvggish import vggish_input

FFT_SIZE_IN_SECS = 0.05
HOP_LENGTH_IN_SECS = 0.01


@cached(thread_safe=False, max_size=100_000)
def _get_spectrogram(filename, max_duration, target_sample_rate, return_mfcc=False):
    try:
        waveform, sample_rate = torchaudio.load(filename)
    except RuntimeError as e:
        import librosa
        waveform, sample_rate = librosa.load(filename, sr=None)
        waveform = torch.tensor(waveform).unsqueeze(0)

    waveform = torch.mean(waveform, dim=0).unsqueeze(0)
    if sample_rate != target_sample_rate:
        transform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = transform(waveform)

    n_fft = int(FFT_SIZE_IN_SECS * target_sample_rate)
    hop_length = int(HOP_LENGTH_IN_SECS * target_sample_rate)

    if waveform.shape[1] < n_fft:
        waveform = F.pad(waveform, (0, n_fft - waveform.shape[1]))

    if return_mfcc:
        transform = torchaudio.transforms.MFCC(
            sample_rate=target_sample_rate,
            n_mfcc=20,
            melkwargs={
                'n_mels': 128,
                'n_fft': n_fft,
                'hop_length': hop_length,
                'power': 2.0})
    else:
        transform = torchaudio.transforms.MelSpectrogram(
            n_mels=128,
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0)
    spec = transform(waveform)

    frames_per_sec = int(1 / HOP_LENGTH_IN_SECS)
    max_frames = int(max_duration * frames_per_sec)

    spec = spec[0, :, :max_frames]
    if spec.shape[1] < max_frames:
        spec = F.pad(spec, (0, max_frames - spec.shape[1]))

    return spec

@cached(thread_safe=False, max_size=100_000)
def _get_waveform(filename, max_duration, target_sample_rate):
    try:
        waveform, sample_rate = torchaudio.load(filename)
    except RuntimeError as e:
        import librosa
        waveform, sample_rate = librosa.load(filename, sr=None)
        waveform = torch.tensor(waveform).unsqueeze(0)

    waveform = torch.mean(waveform, dim=0).unsqueeze(0)

    if sample_rate != target_sample_rate:
        transform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = transform(waveform)

    max_samples = max_duration * target_sample_rate
    waveform = waveform[0, :max_samples]
    if waveform.shape[0] < max_samples:
        waveform = F.pad(waveform, (0, max_samples - waveform.shape[0]))

    return waveform

@cached(thread_safe=False, max_size=100_000)
def _get_vggish_spectrogram(filename, max_duration, target_sample_rate=16_000):
    assert target_sample_rate == 16_000

    waveform = _get_waveform(filename, max_duration, target_sample_rate).numpy()
    spec = vggish_input.waveform_to_examples(waveform, target_sample_rate, return_tensor=True)
    return spec


@cached(thread_safe=False, max_size=100_000)
def _get_vggish_spectrogram_with_offset(filename, st, ed, max_duration, target_sample_rate=16_000):
    assert target_sample_rate == 16_000

    waveform = _get_waveform(filename, max_duration, target_sample_rate).numpy()
    waveform = waveform[st:ed]
    spec = vggish_input.waveform_to_examples(waveform, target_sample_rate, return_tensor=True)
    return spec


class ClassificationDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        num_labels,
        labels,
        unknown_label,
        sample_rate,
        max_duration,
        feature_type):

        super().__init__()

        label_to_id = {lbl: i for i, lbl in enumerate(labels)}
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.feature_type = feature_type

        df = pd.read_csv(metadata_path)

        self.xs = []
        self.ys = []

        for _, row in df.iterrows():
            self.xs.append(row['path'])

            if row['label'] not in label_to_id:
                if unknown_label is not None:
                    label_id = label_to_id[unknown_label]
                else:
                    raise KeyError(f"Unknown label: {row['label']}")
            else:
                label_id = label_to_id[row['label']]

            self.ys.append(label_id)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if self.feature_type == 'waveform':
            x = _get_waveform(
                self.xs[idx],
                max_duration=self.max_duration,
                target_sample_rate=self.sample_rate)

        elif self.feature_type == 'vggish':
            x = _get_vggish_spectrogram(
                self.xs[idx],
                max_duration=self.max_duration)

        elif self.feature_type == 'melspectrogram':
            x = _get_spectrogram(
                self.xs[idx],
                max_duration=self.max_duration,
                target_sample_rate=self.sample_rate)

        elif self.feature_type == 'mfcc':
            x = _get_spectrogram(
                self.xs[idx],
                max_duration=self.max_duration,
                target_sample_rate=self.sample_rate,
                return_mfcc=True)
        else:
            assert False

        return (x, self.ys[idx])


class RecognitionDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        num_labels,
        labels,
        unknown_label,
        sample_rate,
        max_duration,
        window_width,
        window_shift,
        feature_type):

        label_to_id = {lbl: i for i, lbl in enumerate(labels)}
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.feature_type = feature_type

        self.xs = []
        self.ys = []

        if self.feature_type == 'waveform':
            size_per_sec = sample_rate
        elif self.feature_type == 'vggish':
            size_per_sec = 16_000       # fixed sample rate for VGGish
        else:
            size_per_sec = int(1 / HOP_LENGTH_IN_SECS)

        with open(metadata_path) as f:
            for line in f:
                data = json.loads(line)
                wav_path = data['path']
                length = data['length']

                num_windows = int((length - window_width) / window_shift) + 1

                for window_id in range(num_windows):
                    st, ed = window_id * window_shift, window_id * window_shift + window_width
                    offset_st, offset_ed = st * size_per_sec, ed * size_per_sec
                    self.xs.append((wav_path, offset_st, offset_ed))

                    y = torch.zeros(num_labels)

                    for anon in data['annotations']:
                        if anon['label'] not in label_to_id:
                            if unknown_label is not None:
                                label_id = label_to_id[unknown_label]
                            else:
                                raise KeyError(f"Unknown label: {anon['label']}")
                        else:
                            label_id = label_to_id[anon['label']]

                        if (st <= anon['st'] <= ed) or (st <= anon['ed'] <= ed):
                            denom = min(ed - st, anon['ed'] - anon['st'])
                            if denom == 0:
                                continue
                            overlap = (min(ed, anon['ed']) - max(st, anon['st'])) / denom
                            if overlap > .2:
                                y[label_id] = 1
                        if anon['st'] <= st and ed <= anon['ed']:
                            y[label_id] = 1

                    self.ys.append(y)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        wav_path, offset_st, offset_ed = self.xs[idx]
        if self.feature_type == 'waveform':
            x = _get_waveform(
                wav_path,
                max_duration=self.max_duration,
                target_sample_rate=self.sample_rate)
            x = x[offset_st:offset_ed]

        elif self.feature_type == 'vggish':
            x = _get_vggish_spectrogram_with_offset(
                wav_path,
                offset_st, offset_ed,
                max_duration=self.max_duration)

        elif self.feature_type == 'melspectrogram':
            x = _get_spectrogram(
                wav_path,
                max_duration=self.max_duration,
                target_sample_rate=self.sample_rate)
            x = x[:, offset_st:offset_ed]

        elif self.feature_type == 'mfcc':
            x = _get_spectrogram(
                wav_path,
                max_duration=self.max_duration,
                target_sample_rate=self.sample_rate,
                return_mfcc=True)
            x = x[:, offset_st:offset_ed]

        return x, self.ys[idx]
