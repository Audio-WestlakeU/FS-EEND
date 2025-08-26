#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Modified by: Di Liang
# Licensed under the MIT license.
#
import torch
import numpy as np
from .kaldi_data import KaldiData
from .feature import *
from glob import glob
import soundfile as sf

def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size=2000, step=2000,
        use_last_samples=False,
        label_delay=0,
        subsampling=1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - label_delay > 0:
            yield (i + 1) * step, data_length


def my_collate(batch):
    data, rec = list(zip(*batch))
    return [data, rec]


class KaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            chunk_size=2000,
            context_size=0,
            frame_size=1024,
            frame_shift=256,
            subsampling=1,
            rate=16000,
            input_transform=None,
            use_last_samples=False,
            label_delay=0,
            n_speakers=None,
            shuffle=False
            ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.chunk_indices = []
        self.label_delay = label_delay
        self.shuffle = shuffle

        self.wavs = glob(f"{data_dir}/**/*.flac", recursive=True) + glob(f"{data_dir}/**/*.wav", recursive=True)

        # make chunk indices: filepath, start_frame, end_frame
        print(len(self.wavs), " wavs")

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, i):
        wav_path = self.wavs[i]
        rec = wav_path.split('/')[-1].split('.')[0]
        data, rate = sf.read(wav_path, dtype='float32')
        Y = stft(data, self.frame_size, self.frame_shift)
        # Y: (frame, num_ceps)
        Y = transform(Y, self.input_transform)
        # Y_spliced: (frame, num_ceps * (context_size * 2 + 1))
        Y_spliced = splice(Y, self.context_size)
        # Y_ss: (frame / subsampling, num_ceps * (context_size * 2 + 1))
        Y_ss = Y_spliced[::self.subsampling]

        Y_ss = torch.from_numpy(Y_ss).float()
        
        if self.shuffle:
            order = np.arange(Y_ss.shape[0])
            np.random.shuffle(order)
            Y_ss = Y_ss[order]
        
        return Y_ss, rec
