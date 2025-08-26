import tqdm
import torch
from torch import Tensor
import numpy as np
import torch.nn.functional as F
from numpy import random as nr
import random

from scipy.optimize import linear_sum_assignment

from torch.utils.data import WeightedRandomSampler


def find_enroll_singl_spk_frames(feats, preds: Tensor, decis: Tensor, utt_floor: int):
    '''
    input:
        decis : (T, S + 2) predicted speaker voice activity
        utt_floor: the mininum length of an utterance
    output:

    '''
    single_spk_idx = decis.sum(1) == 1
    decis_single_spk_frames = decis[single_spk_idx]
    feats_single_spk_frames = feats[single_spk_idx]
    preds_single_spk_frames = preds[single_spk_idx]
    spk_x = []
    spk_y = []

    for spkid, frames in enumerate(decis_single_spk_frames.T):
        frames = F.pad(frames, (1, 1), 'constant')
        changes, = torch.where(torch.diff(frames, dim=0) != 0)
        enroll_i_idx_set = []
        enroll_i_len_set = []
        # 将符合最低长度要求的single-spk utterance挑选出来
        for s, e in zip(changes[::2], changes[1::2]):
            if e - s >= utt_floor:
                idx = list(range(s, e))
                enroll_i_idx_set.append(idx)
                enroll_i_len_set.append(e - s)
        # single-spk utterance with the maximum length, as the enroll segment
        if len(enroll_i_idx_set) > 0:
            seg = enroll_i_idx_set[enroll_i_len_set.index(max(enroll_i_len_set))]
            if spkid == 0:  # silence segment
                silen_x = feats_single_spk_frames[seg]
                silen_y = preds_single_spk_frames[seg]
            else:           # spk segment
                spk_x.append(feats_single_spk_frames[seg])
                spk_y.append(preds_single_spk_frames[seg])
                    
    # 合成注册段（包括输入特征x和预测值y）
    enroll_x = silen_x
    enroll_y = silen_y
    for x, y in zip(spk_x, spk_y):
        enroll_x = torch.cat([enroll_x, x], dim=0)
        enroll_y = torch.cat([enroll_y, y], dim=0)
        enroll_x = torch.cat([enroll_x, silen_x], dim=0)
        enroll_y = torch.cat([enroll_y, silen_y], dim=0)

    return enroll_x, enroll_y


def find_enroll_segment(decis, utt_floor):
    enroll_spkid = []
    enroll_idx = []
    for spkid, frames in enumerate(decis.T):
        frames = F.pad(frames, (1, 1), 'constant')
        changes, = torch.where(torch.diff(frames, dim=0) != 0)
        for s, e in zip(changes[::2], changes[1::2]):
            if e - s >= utt_floor:
                idx = list(range(s,e))
                enroll_idx += idx
                enroll_spkid.append(spkid)
                break
    
    return enroll_idx, enroll_spkid, len(enroll_spkid)


def select_sigl_spk_frames(decis: Tensor, enroll_idx: list, mod_frame: int) -> list:
    decis_resi = decis.clone()
    decis_resi[enroll_idx] = 0
    decis_res_sigl_idx = (decis_resi.sum(1) == 1).nonzero().squeeze().tolist()

    # Single spk frames
    sigl_spk_idx = []
    for spkid, frames in enumerate(decis_resi.T):
        spk_i_idx = (frames > 0).nonzero().squeeze(1).tolist()
        # spk_i_sigl_idx = list(set(decis_res_sigl_idx) & set(spk_i_idx))
        spk_i_sigl_idx = list(set(spk_i_idx))
        spk_i_sigl_idx.sort()
        # random.shuffle(spk_i_sigl_idx)
        sigl_spk_idx += spk_i_sigl_idx[:mod_frame]
    
    sigl_spk_idx = list(set(sigl_spk_idx))
    
    return sigl_spk_idx

def split_inp(feats: list[Tensor], labels: list[Tensor], T_prime: int):
    # fea: (T, D)
    rechunk_feats = []
    rechunk_labels = []
    for fea, l in zip(feats, labels):
        rechk_fea = list(fea.split(T_prime, dim=0))
        rechunk_feats = rechunk_feats + rechk_fea
        rechk_l = list(l.split(T_prime, dim=0))
        rechunk_labels = rechunk_labels + rechk_l
    return rechunk_feats, rechunk_labels


def resize_chunk(T: int)->int:
    var_chunks = np.array([50, 100, 200, 500, 1000])
    T_prime = T
    if torch.rand(1) >= 0.5:
        T_prime = min(nr.choice(var_chunks, size=(1), replace=False).tolist()[0], T)
    return T_prime


def upd_buf_ver2(x_buf, x_i, z_buf, y_i, buf_size):
    # (T, S_i)
    x_cat = torch.cat([x_buf, x_i], dim=0)
    y_cat = torch.cat([z_buf, y_i], dim=0)

    T, S_i = y_cat.shape
    p = y_cat / y_cat.sum(dim=1, keepdim=True)
    p = p.masked_fill_(p == 0, 1e-6)
    q = torch.tensor(1.0 / S_i).repeat(T, S_i).to(p)
    r = torch.sum(y_cat / y_cat.sum(dim=0, keepdim=True), dim=1)
    KLD = r * ((p * torch.log(p / q)).sum(dim=1))

    prob = KLD / KLD.sum(dim=0, keepdim=True)
    # prob_cumsum = F.pad(prob.cumsum(dim=0), (1, 0), mode='constant', value=0.)

    # t_selct = []
    # for i in range(buf_size):
    #     rand_prob = torch.rand(1).to(prob_cumsum)
    #     for t in range(1, T + 1):
    #         if rand_prob >= prob_cumsum[t-1] and rand_prob <= prob_cumsum[t]:
    #             t_selct.append(t - 1)
    t_selct = list(WeightedRandomSampler(prob, buf_size, replacement=False))
    t_selct.sort()
    x_buf_upd = x_cat[t_selct]
    y_buf_upd = y_cat[t_selct]

    return x_buf_upd, y_buf_upd

def upd_buf(x_buf, x_i, z_buf, y_i, buf_size):
    # (T, S_i)
    x_cat = torch.cat([x_buf, x_i], dim=0)
    y_cat = torch.cat([z_buf, y_i], dim=0)

    T, S_i = y_cat.shape
    p = y_cat / y_cat.sum(dim=1, keepdim=True)
    p = p.masked_fill_(p == 0, 1e-6)
    q = torch.tensor(1.0 / S_i).repeat(T, S_i).to(p)
    KLD =  (p * torch.log(p / q)).sum(dim=1)
    KLD = KLD.masked_fill_(KLD<0, 0)
    KLD = KLD.masked_fill_(KLD==0, 1e-6)

    prob = KLD / KLD.sum(dim=0, keepdim=True)
    # prob_cumsum = F.pad(prob.cumsum(dim=0), (1, 0), mode='constant', value=0.)

    # t_selct = []
    # for i in range(buf_size):
    #     rand_prob = torch.rand(1).to(prob_cumsum)
    #     for t in range(1, T + 1):
    #         if rand_prob >= prob_cumsum[t-1] and rand_prob <= prob_cumsum[t]:
    #             t_selct.append(t - 1)
    t_selct = list(WeightedRandomSampler(prob, buf_size, replacement=False))
    t_selct.sort()
    x_buf_upd = x_cat[t_selct]
    y_buf_upd = y_cat[t_selct]

    return x_buf_upd, y_buf_upd


def upd_buf_FIFO(x_buf, x_i, z_buf, y_i, buf_size):
    # (T, S_i)
    x_cat = torch.cat([x_buf, x_i], dim=0)
    y_cat = torch.cat([z_buf, y_i], dim=0)

    x_len = x_cat.shape[0]

    # print(x_len - buf_size, x_cat[x_len-buf_size:].shape, y_cat[x_len-buf_size:].shape)

    return x_cat[x_len-buf_size:], y_cat[x_len-buf_size:]

def cal_cor(A: Tensor, B: Tensor):
    # (T)
    A_mean = A.mean()
    B_mean = B.mean()
    cov = torch.sum((A - A_mean) * (B - B_mean))
    stdv1 = torch.sqrt(torch.sum((A - A_mean) * (A - A_mean)))
    stdv2 = torch.sqrt(torch.sum((B - B_mean) * (B - B_mean)))
    return cov / (stdv1 * stdv2 + 1e-6)



def find_best_perm(y: Tensor, y_pred: Tensor):
    # y(T, C) # y_pred (T, C)
    T, C = y.shape
    cc_mtrx = torch.zeros(C, C)
    row = torch.arange(C)
    for i in range(y.shape[1]):
        col = row.roll(shifts=i, dims=0)
        for ro, co in zip(row, col):
            cc_mtrx[ro, co] = cal_cor(y[:, ro], y_pred[:, co])
    
    best_perm = linear_sum_assignment(cc_mtrx, True)[1]
    return best_perm


class ContextBuilder(torch.nn.Module):
    def __init__(self, context_size) -> None:
        super(ContextBuilder, self).__init__()
        self.context_size = context_size

    def forward(self, x):
        bsz, T, _ = x.shape
        x_pad = torch.nn.functional.pad(x, (0, 0, self.context_size, self.context_size))
        return x_pad.unfold(1, T, 1).reshape(bsz, -1, T).transpose(-1, -2)


class TorchScaler(torch.nn.Module):
    """
    This torch module implements scaling for input tensors, both instance based
    and dataset-wide statistic based.

    Args:
        statistic: str, (default='dataset'), represent how to compute the statistic for normalisation.
            Choice in {'dataset', 'instance'}.
             'dataset' needs to be 'fit()' with a dataloader of the dataset.
             'instance' apply the normalisation at an instance-level, so compute the statitics on the instance
             specified, it can be a clip or a batch.
        normtype: str, (default='standard') the type of normalisation to use.
            Choice in {'standard', 'mean', 'minmax'}. 'standard' applies a classic normalisation with mean and standard
            deviation. 'mean' substract the mean to the data. 'minmax' substract the minimum of the data and divide by
            the difference between max and min.
    """

    def __init__(self, statistic="dataset", normtype="standard", dims=(1, 2), eps=1e-8):
        super(TorchScaler, self).__init__()
        assert statistic in ["dataset", "instance"]
        assert normtype in ["standard", "mean", "minmax"]
        if statistic == "dataset" and normtype == "minmax":
            raise NotImplementedError(
                "statistic==dataset and normtype==minmax is not currently implemented."
            )
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps

    def load_state_dict(self, state_dict, strict=True):
        if self.statistic == "dataset":
            super(TorchScaler, self).load_state_dict(state_dict, strict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.statistic == "dataset":
            super(TorchScaler, self)._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def fit(self, dataloader, transform_func=lambda x: x[0]):
        """
        Scaler fitting

        Args:
            dataloader (DataLoader): training data DataLoader
            transform_func (lambda function, optional): Transforms applied to the data.
                Defaults to lambdax:x[0].
        """
        indx = 0
        for batch in tqdm.tqdm(dataloader):

            feats = transform_func(batch)
            if indx == 0:
                mean = torch.mean(feats, self.dims, keepdim=True).mean(0).unsqueeze(0)
                mean_squared = (
                    torch.mean(feats ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                )
            else:
                mean += torch.mean(feats, self.dims, keepdim=True).mean(0).unsqueeze(0)
                mean_squared += (
                    torch.mean(feats ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                )
            indx += 1

        mean /= indx
        mean_squared /= indx

        self.register_buffer("mean", mean)
        self.register_buffer("mean_squared", mean_squared)

    def forward(self, tensor):
        if self.statistic == "dataset":
            assert hasattr(self, "mean") and hasattr(
                self, "mean_squared"
            ), "TorchScaler should be fit before used if statistics=dataset"
            assert tensor.ndim == self.mean.ndim, "Pre-computed statistics "
            if self.normtype == "mean":
                return tensor - self.mean
            elif self.normtype == "standard":
                std = torch.sqrt(self.mean_squared - self.mean ** 2)
                return (tensor - self.mean) / (std + self.eps)
            else:
                raise NotImplementedError

        else:
            if self.normtype == "mean":
                return tensor - torch.mean(tensor, self.dims, keepdim=True)
            elif self.normtype == "standard":
                return (tensor - torch.mean(tensor, self.dims, keepdim=True)) / (
                    torch.std(tensor, self.dims, keepdim=True) + self.eps
                )
            elif self.normtype == "minmax":
                return (tensor - torch.amin(tensor, dim=self.dims, keepdim=True)) / (
                    torch.amax(tensor, dim=self.dims, keepdim=True)
                    - torch.amin(tensor, dim=self.dims, keepdim=True)
                    + self.eps
                )

def splice(Y, context_size=0):
    """ Frame splicing

    Args:
        Y: feature
            (n_frames, n_featdim)-shaped numpy array
        context_size:
            number of frames concatenated on left-side
            if context_size = 5, 11 frames are concatenated.

    Returns:
        Y_spliced: spliced feature
            (n_frames, n_featdim * (2 * context_size + 1))-shaped
    """
    Y_pad = np.pad(
            Y,
            [(context_size, context_size), (0, 0)],
            'constant')
    Y_spliced = np.lib.stride_tricks.as_strided(
            Y_pad,
            (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
            (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
    return Y_spliced

# if __name__ == "__main__":
#     import numpy as np

#     context = ContextBuilder(7)
#     data = torch.rand([1, 10, 2])
#     data_contx = context(data)

#     data_np = data.numpy()[0]
#     data_contx_np = splice(data_np, 7)

#     print(data_contx.numpy()[0] == data_contx_np)

if __name__=="__main__":
    a = torch.tensor([[1.0,2.0], [3.0,4.0], [2.0,1.0]])
    b = torch.tensor([[1.0,1.0], [2.0,4.0], [2.0,2.0]])
    find_best_perm(a, b)