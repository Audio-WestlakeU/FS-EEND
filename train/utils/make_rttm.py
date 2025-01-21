import argparse
import h5py
import numpy as np
import os
from scipy.signal import medfilt
import torch
import torch.nn.functional as F

def make_rttm(rec, pred, frame_shift=80, threshold=0.5, median=11, subsampling=10, sampling_rate=8000, out_rttm_file=None):
    with open(out_rttm_file, 'a') as wf:
        pred = torch.where(pred > threshold, 1, 0)
        if median > 1:
            pred = medfilt(pred, (median, 1))
            pred = torch.from_numpy(pred).float()

        for spkid, frames in enumerate(pred.T):
            frames = F.pad(frames, (1, 1), 'constant')
            changes, = torch.where(torch.diff(frames, dim=0) != 0)
            fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
            for s, e in zip(changes[::2], changes[1::2]):
                print(fmt.format(
                    rec,
                    s * frame_shift * subsampling / sampling_rate,
                    (e - s) * frame_shift * subsampling / sampling_rate,
                    rec + "_" + str(spkid)), file=wf)
