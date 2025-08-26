#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os
import torch
import numpy as np
from .kaldi_data import KaldiData
from .feature import *

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
    data, target, rec = list(zip(*batch))
    return [data, target, rec]


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

        self.data = KaldiData(self.data_dir)

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            data_len = int(self.data.reco2dur[rec] * rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            for st, ed in _gen_frame_indices(
                    data_len, chunk_size, chunk_size, use_last_samples,
                    label_delay=self.label_delay,
                    subsampling=self.subsampling):
                self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))
        print(len(self.chunk_indices), " chunks")

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        rec, st, ed = self.chunk_indices[i]
        Y, T = get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers)
        # Y: (frame, num_ceps)
        Y = transform(Y, self.input_transform, rec=rec)
        # Y_spliced: (frame, num_ceps * (context_size * 2 + 1))
        Y_spliced = splice(Y, self.context_size)
        # Y_ss: (frame / subsampling, num_ceps * (context_size * 2 + 1))
        Y_ss, T_ss = subsample(Y_spliced, T, self.subsampling)

        Y_ss = torch.from_numpy(Y_ss).float()
        T_ss = torch.from_numpy(T_ss).float()
        
        if self.shuffle:
            order = np.arange(Y_ss.shape[0])
            np.random.shuffle(order)
            Y_ss = Y_ss[order]
            T_ss = T_ss[order]
        # save Y
        save_path = "/mnt/home/liangdi/projects/pl_version/pl_eend/tsne_visual/figures/onl_2spk_version_tfm_10w_ver_266/Y_aishell_1_mix_0000001_input_flac.npy"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if rec == "data_simu_HQ_aishell_1_simu_2spk_8k_wav_HQ_aishell_1_8k_cv_ns2_beta2_500_1_mix_0000001":
            # print("cummn_logmel23: ", Y)
            print("statics of cummn_logmel23: ", np.mean(Y), np.std(Y), np.max(Y), np.min(Y), Y.shape)
            np.save(save_path, Y)
        
        return Y_ss, T_ss, rec
    
    def __getfulllabel__(self, i):
        rec, st, ed = self.chunk_indices[i]
        Y, T = get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers)
        # Y: (frame, num_ceps)
        Y = transform(Y, self.input_transform)
        # Y_spliced: (frame, num_ceps * (context_size * 2 + 1))
        Y_spliced = splice(Y, self.context_size)
        # Y_ss: (frame / subsampling, num_ceps * (context_size * 2 + 1))
        Y_ss, T_ss = subsample(Y_spliced, T, self.subsampling)
        T_ss = torch.from_numpy(T_ss).float()
        T = torch.from_numpy(T).float()
        return T, rec
    
    def __get_len__(self, i):
        rec, st, ed = self.chunk_indices[i]
        Y, T = get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers)
        T = torch.from_numpy(T).float()
        return T.shape[0]
    
