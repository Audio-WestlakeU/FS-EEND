# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Modified by: Di Liang
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment
from torchmetrics import PermutationInvariantTraining
from collections import defaultdict

from torch import Tensor
"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""

# labels: [T * C] - T: 50s

def split_results(labels: list[Tensor], preds: list[Tensor], ilens: list[int], blk_size: int):
    n_speakers_split = []
    labels_splt = []
    preds_splt = []
    for l, p, ilen in zip(labels, preds, ilens):
        n_blk = len(l.split(blk_size, dim=0))
        for i in range(n_blk):
            st = i * blk_size
            ed = min((i + 1) * blk_size, ilen)
            non_zero_spk = torch.max(l[:ed, :], dim=0)[0]
            non_zero_spk_idx = non_zero_spk > 0
            nspk = int(non_zero_spk.sum())
            n_speakers_split.append(nspk)
            labels_splt.append(l[st:ed, non_zero_spk_idx])
            preds_splt.append(p[st:ed, :nspk])
    return labels_splt, preds_splt, n_speakers_split

def pad_labels(ts, out_size):
    # pad label's speaker-dim to be model's n_speakers
    for i, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding
            ts[i] = F.pad(t, (0, out_size - t.shape[1], 0, 0), mode='constant', value=0.)
        elif t.shape[1] > out_size:
            # truncate
            raise ValueError
    return ts


def pad_preds(ys, out_size):
    # pad label's speaker-dim to be model's n_speakers
    ys_padded = []
    for i, y in enumerate(ys):
        if y.shape[1] < out_size:
            # padding
            ys_padded.append(torch.cat([y, torch.zeros((y.shape[0], out_size - y.shape[1]), dtype=y.dtype, device=y.device)], dim=1))
        elif y.shape[1] > out_size:
            # truncate
            raise ValueError
        else:
            ys_padded.append(y)
    return ys_padded

def pit_loss(pred, label, label_delay=0):
    """
    Permutation-invariant training (PIT) cross entropy loss function.

    Args:
      pred:  (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
            pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
    """
    # label permutations along the speaker axis
    label_perms = [label[..., list(p)] for p
                    in permutations(range(label.shape[-1]))]
    losses = torch.stack(
        [F.binary_cross_entropy_with_logits(
            pred[label_delay:, ...],
            l[:len(l) - label_delay, ...]) for l in label_perms])
    min_loss = losses.min() * (len(label) - label_delay)
    min_index = losses.argmin().detach()
    
    return min_loss, label_perms[min_index]


def batch_pit_loss(ys, ts, label_delay=0):
    """
    PIT loss over mini-batch.

    Args:
      ys: B-length list of predictions
      ts: B-length list of labels

    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    loss_w_labels = [pit_loss(y, t, label_delay)
                     for (y, t) in zip(ys, ts)]
    losses, labels = zip(*loss_w_labels)
    loss = torch.stack(losses).sum()
    n_frames = np.sum([t.shape[0] for t in ts])
    loss = loss / n_frames
    return loss, labels


def standard_loss(ys, ts, label_delay=0):
    losses = [F.binary_cross_entropy_with_logits(y[label_delay:, ...], t[:len(t) - label_delay, ...]) * (len(y) - label_delay)
                for (y, t) in zip(ys, ts)]
    loss = torch.stack(losses).sum()
    n_frames = np.sum([t.shape[0] for t in ts]) - (label_delay * len(ts))
    loss = loss / n_frames
    return loss

def standard_mask_loss(ys, ts, label_delay=0):
    losses = []
    for (y, t) in zip(ys, ts):
        loss_mtrx = F.binary_cross_entropy_with_logits(y[label_delay:, ...], t[:len(t) - label_delay, ...], reduce=False)
        loss_mtrx = loss_mtrx.masked_fill((t == 0) & (y < 0), 0)
        loss = loss_mtrx.mean() * (len(y) - label_delay)
        losses.append(loss)
    loss = torch.stack(losses).sum()
    n_frames = np.sum([t.shape[0] for t in ts]) - (label_delay * len(ts))
    loss = loss / n_frames
    return loss


def calc_diarization_error2(pred, label, label_delay=0):
    """
    Calculates diarization error stats for reporting.

    Args:
      pred (torch.FloatTensor): (T,C)-shaped pre-activation values
      label (torch.FloatTensor): (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      res: dict of diarization error stats
    """
    label = label[:len(label) - label_delay, ...]
    decisions = pred[label_delay:, ...] > 0.5
    n_ref = label.sum(axis=-1).long()
    n_sys = decisions.sum(axis=-1).long()
    res = {}
    res['speech_scored'] = (n_ref > 0).sum()
    res['speech_miss'] = ((n_ref > 0) & (n_sys == 0)).sum()
    res['speech_falarm'] = ((n_ref == 0) & (n_sys > 0)).sum()
    res['speaker_scored'] = (n_ref).sum()
    res['speaker_miss'] = torch.max((n_ref - n_sys), torch.zeros_like(n_ref)).sum()
    res['speaker_falarm'] = torch.max((n_sys - n_ref), torch.zeros_like(n_ref)).sum()
    n_map = ((label == 1) & (decisions == 1)).sum(axis=-1)
    res['speaker_error'] = (torch.min(n_ref, n_sys) - n_map).sum()
    res['correct'] = (label == decisions).sum() / label.shape[1]
    res['diarization_error'] = (
        res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
    res['frames'] = len(label)
    # print(res['diarization_error'] / res['speech_scored'], res['speaker_error'] / res['speech_scored'], res['speaker_falarm'] / res['speech_scored'], res['speaker_miss'] / res['speech_scored'])
    # if res['speaker_error'] < 0:
    #     print(decisions)
    #     print(label)
    #     raise Exception("spk error")
    return res



def report_diarization_error2(ys, labels, label_delay=0):
    """
    Reports diarization errors
    Should be called with torch.no_grad

    Args:
      ys: B-length list of predictions (torch.FloatTensor)
      labels: B-length list of labels (torch.FloatTensor)
    """
    stats_batch = defaultdict(list)
    for i, (y, t) in enumerate(zip(ys, labels)):
        stats = calc_diarization_error2(y, t, label_delay)
        for k, v in stats.items():
            stats_batch[k].append(float(v))

    return stats_batch

def calc_diarization_error(pred, label, label_delay=0):
    """
    Calculates diarization error stats for reporting.

    Args:
      pred (torch.FloatTensor): (T,C)-shaped pre-activation values
      label (torch.FloatTensor): (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      res: dict of diarization error stats
    """
    label = label[:len(label) - label_delay, ...]
    decisions = torch.sigmoid(pred[label_delay:, ...]) > 0.5
    n_ref = label.sum(axis=-1).long()
    n_sys = decisions.sum(axis=-1).long()
    res = {}
    res['speech_scored'] = (n_ref > 0).sum()
    res['speech_miss'] = ((n_ref > 0) & (n_sys == 0)).sum()
    res['speech_falarm'] = ((n_ref == 0) & (n_sys > 0)).sum()
    res['speaker_scored'] = (n_ref).sum()
    res['speaker_miss'] = torch.max((n_ref - n_sys), torch.zeros_like(n_ref)).sum()
    res['speaker_falarm'] = torch.max((n_sys - n_ref), torch.zeros_like(n_ref)).sum()
    n_map = ((label == 1) & (decisions == 1)).sum(axis=-1)
    res['speaker_error'] = (torch.min(n_ref, n_sys) - n_map).sum()
    res['correct'] = (label == decisions).sum() / label.shape[1]
    res['diarization_error'] = (
        res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
    res['frames'] = len(label)
    # print(res['diarization_error'] / res['speech_scored'], res['speaker_error'] / res['speech_scored'])
    # print(res['diarization_error'] / res['speaker_scored'])
    if res['speaker_error'] < 0:
        print(decisions)
        print(label)
        raise Exception("spk error")
    return res


def report_diarization_error(ys, labels, label_delay=0):
    """
    Reports diarization errors
    Should be called with torch.no_grad

    Args:
      ys: B-length list of predictions (torch.FloatTensor)
      labels: B-length list of labels (torch.FloatTensor)
    """
    stats_batch = defaultdict(list)
    for i, (y, t) in enumerate(zip(ys, labels)):
        stats = calc_diarization_error(y, t, label_delay)
        for k, v in stats.items():
            stats_batch[k].append(float(v))

    return stats_batch


def batch_pit_n_speaker_loss(ys, ts, n_speakers_list):
    """
    PIT loss over mini-batch.
    Args:
      ys: B-length list of predictions (pre-activations)
      ts: B-length list of labels
      n_speakers_list: list of n_speakers in batch
    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    max_n_speakers = max(n_speakers_list)
    # (B, T, C)
    ys = nn.utils.rnn.pad_sequence(ys, padding_value=-1, batch_first=True)

    losses = []
    for shift in range(max_n_speakers):
        # rolled along with speaker-axis
        ts_roll = [torch.roll(t, -shift, dims=1) for t in ts]
        ts_roll = nn.utils.rnn.pad_sequence(ts_roll, padding_value=-1, batch_first=True)
        # loss: (B, T, C)
        loss = F.binary_cross_entropy_with_logits(ys, ts_roll, reduce=False)
        # sum over time: (B, C)
        loss = torch.sum(loss, dim=1)
        losses.append(loss)
    # losses: (B, C, C)
    losses = torch.stack(losses, dim=2)
    # losses[b, i, j] is a loss between
    # `i`-th speaker in y and `(i+j)%C`-th speaker in t
    
    perms = np.array(
        list(permutations(range(max_n_speakers))),
        dtype='i',
    )
    # y_ind: [0,1,2,3]
    y_ind = np.arange(max_n_speakers, dtype='i')
    #  perms  -> relation to t_inds      -> t_inds
    # 0,1,2,3 -> 0+j=0,1+j=1,2+j=2,3+j=3 -> 0,0,0,0
    # 0,1,3,2 -> 0+j=0,1+j=1,2+j=3,3+j=2 -> 0,0,1,3
    t_inds = np.mod(perms - y_ind, max_n_speakers)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(
            torch.mean(losses[:, y_ind, t_ind], dim=1))
    # losses_perm: (B, Perm)
    losses_perm = torch.stack(losses_perm, dim=1)    

    # masks: (B, Perms)
    def select_perm_indices(num, max_num):
        perms = list(permutations(range(max_num)))
        sub_perms = list(permutations(range(num)))
        return [
            [x[:num] for x in perms].index(perm)
            for perm in sub_perms]
    masks = torch.full_like(losses_perm, np.inf)
    for i, t in enumerate(ts):
        n_speakers = n_speakers_list[i]
        indices = select_perm_indices(n_speakers, max_n_speakers)
        masks[i, indices] = 0
    losses_perm += masks

    min_loss = torch.sum(torch.min(losses_perm, dim=1)[0])
    n_frames = np.sum([t.shape[0] for t in ts])
    min_loss = min_loss / n_frames

    min_indices = torch.argmin(losses_perm, dim=1)
    labels_perm = [t[:, perms[idx]] for t, idx in zip(ts, min_indices)]
    labels_perm = [t[:, :n_speakers] for t, n_speakers in zip(labels_perm, n_speakers_list)]

    return min_loss, labels_perm

def batch_pit_n_speaker_loss_label_delay(ys, ts, n_speakers_list, label_delay=0):
    """
    PIT loss over mini-batch.
    Args:
      ys: B-length list of predictions (pre-activations)
      ts: B-length list of labels
      n_speakers_list: list of n_speakers in batch
    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    max_n_speakers = max(n_speakers_list)
    ilens = [t.shape[0] for t in ts]
    # (B, T, C)
    ys = nn.utils.rnn.pad_sequence(ys, padding_value=-1, batch_first=True)
    B, T, C = ys.shape


    losses = []
    for shift in range(max_n_speakers):
        # rolled along with speaker-axis
        ts_roll = [torch.roll(t, -shift, dims=1) for t in ts]
        ts_roll = nn.utils.rnn.pad_sequence(ts_roll, padding_value=-1, batch_first=True)
        # loss: (B, T, C)
        loss = F.binary_cross_entropy_with_logits(ys[:, label_delay:, :], ts_roll[:, :T - label_delay, :], reduce=False)
        # sum over time: (B, C)
        for i, ilen in enumerate(ilens):
            loss[i, ilen-label_delay:, :] = 0
        loss = torch.sum(loss, dim=1)
        losses.append(loss)
    # losses: (B, C, C)
    losses = torch.stack(losses, dim=2)
    # losses[b, i, j] is a loss between
    # `i`-th speaker in y and `(i+j)%C`-th speaker in t
    
    perms = np.array(
        list(permutations(range(max_n_speakers))),
        dtype='i',
    )
    # y_ind: [0,1,2,3]
    y_ind = np.arange(max_n_speakers, dtype='i')
    #  perms  -> relation to t_inds      -> t_inds
    # 0,1,2,3 -> 0+j=0,1+j=1,2+j=2,3+j=3 -> 0,0,0,0
    # 0,1,3,2 -> 0+j=0,1+j=1,2+j=3,3+j=2 -> 0,0,1,3
    t_inds = np.mod(perms - y_ind, max_n_speakers)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(
            torch.mean(losses[:, y_ind, t_ind], dim=1))
    # losses_perm: (B, Perm)
    losses_perm = torch.stack(losses_perm, dim=1)    

    # masks: (B, Perms)
    def select_perm_indices(num, max_num):
        perms = list(permutations(range(max_num)))
        sub_perms = list(permutations(range(num)))
        return [
            [x[:num] for x in perms].index(perm)
            for perm in sub_perms]
    masks = torch.full_like(losses_perm, np.inf)
    for i, t in enumerate(ts):
        n_speakers = n_speakers_list[i]
        indices = select_perm_indices(n_speakers, max_n_speakers)
        masks[i, indices] = 0
    losses_perm += masks

    min_loss = torch.sum(torch.min(losses_perm, dim=1)[0])
    n_frames = np.sum([t.shape[0] for t in ts])
    min_loss = min_loss / n_frames

    min_indices = torch.argmin(losses_perm, dim=1)
    labels_perm = [t[:, perms[idx]] for t, idx in zip(ts, min_indices)]
    labels_perm = [t[:, :n_speakers] for t, n_speakers in zip(labels_perm, n_speakers_list)]

    return min_loss, labels_perm

def batch_pit_n_speaker_loss2(ys, ts, n_speakers_list):
    """
    PIT loss over mini-batch.
    Args:
      ys: B-length list of predictions (pre-activations)
      ts: B-length list of labels
      n_speakers_list: list of n_speakers in batch
    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    max_n_speakers = max(n_speakers_list)
    # (B, T, C)
    ys = nn.utils.rnn.pad_sequence(ys, padding_value=-1, batch_first=True)

    losses = []
    for shift in range(max_n_speakers):
        # rolled along with speaker-axis
        ts_roll = [torch.roll(t, -shift, dims=1) for t in ts]
        ts_roll = nn.utils.rnn.pad_sequence(ts_roll, padding_value=-1, batch_first=True)
        # loss: (B, T, C)
        loss = F.binary_cross_entropy(ys, ts_roll, reduce=False)
        # sum over time: (B, C)
        loss = torch.sum(loss, dim=1)
        losses.append(loss)
    # losses: (B, C, C)
    losses = torch.stack(losses, dim=2)
    # losses[b, i, j] is a loss between
    # `i`-th speaker in y and `(i+j)%C`-th speaker in t
    
    perms = np.array(
        list(permutations(range(max_n_speakers))),
        dtype='i',
    )
    # y_ind: [0,1,2,3]
    y_ind = np.arange(max_n_speakers, dtype='i')
    #  perms  -> relation to t_inds      -> t_inds
    # 0,1,2,3 -> 0+j=0,1+j=1,2+j=2,3+j=3 -> 0,0,0,0
    # 0,1,3,2 -> 0+j=0,1+j=1,2+j=3,3+j=2 -> 0,0,1,3
    t_inds = np.mod(perms - y_ind, max_n_speakers)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(
            torch.mean(losses[:, y_ind, t_ind], dim=1))
    # losses_perm: (B, Perm)
    losses_perm = torch.stack(losses_perm, dim=1)    

    # masks: (B, Perms)
    def select_perm_indices(num, max_num):
        perms = list(permutations(range(max_num)))
        sub_perms = list(permutations(range(num)))
        return [
            [x[:num] for x in perms].index(perm)
            for perm in sub_perms]
    masks = torch.full_like(losses_perm, np.inf)
    for i, t in enumerate(ts):
        n_speakers = n_speakers_list[i]
        indices = select_perm_indices(n_speakers, max_n_speakers)
        masks[i, indices] = 0
    losses_perm += masks

    min_loss = torch.sum(torch.min(losses_perm, dim=1)[0])
    n_frames = np.sum([t.shape[0] for t in ts])
    min_loss = min_loss / n_frames

    min_indices = torch.argmin(losses_perm, dim=1)
    labels_perm = [t[:, perms[idx]] for t, idx in zip(ts, min_indices)]
    labels_perm = [t[:, :n_speakers] for t, n_speakers in zip(labels_perm, n_speakers_list)]

    return min_loss, labels_perm

class PITLoss(nn.Module):
    def __init__(self, n_spks=4) -> None:
        super(PITLoss, self).__init__()
        self.n_spks = n_spks
        trav_idx = []
        for x in permutations(list(range(n_spks))):
            trav_idx += x
        self.trav_idx = torch.tensor(trav_idx).long()
    
    def forward(self, preds, labels):
        B, T, _ = preds.shape

        labels_all_case = labels[:, :, self.trav_idx].reshape(B, T, -1, self.n_spks)
        case_num = labels_all_case.shape[-2]
        preds_all_case = preds.unsqueeze(-1).repeat(1, 1, 1, case_num).transpose(-1, -2)

        loss_all_case = F.binary_cross_entropy_with_logits(preds_all_case, labels_all_case, reduction="none").mean(-1).mean(1)

        min_idx = torch.argmin(loss_all_case, dim=-1)
        selected_loss = loss_all_case[torch.arange(B), min_idx]
        pit_min_loss = selected_loss.sum() / B

        perm_labels = labels_all_case[torch.arange(B), :, min_idx, :]
        return pit_min_loss, perm_labels

    def swap(self, a, i, j):
        temp = a[i]
        a[i] = a[j]
        a[j] = temp

    def dfs(self, a, depth=0):
        r = []
        if depth == len(a):
            r += a
        for i in range(depth, len(a)):
            self.swap(a, i, depth)
            r += self.dfs(a, depth + 1)
            self.swap(a, depth, i)
        return r

if __name__ == "__main__":
    torch.random.manual_seed(2)
    pred = torch.rand([5, 1000, 4]).float()
    label = torch.randint(0, 2, [5, 1000, 4]).float()
    x_1, y_1 = batch_pit_loss(pred, label)

    x_2, y_2 = batch_pit_n_speaker_loss(pred, label)
    print("X:", x_1.item() == x_2.item())
    y_1_t = torch.stack(y_1)
    y_2_t = torch.stack(y_2)
    print("Y:", (y_1_t == y_2_t).float().mean())

    print(report_diarization_error(y_1, label))
    print(report_diarization_error(y_2, label))

