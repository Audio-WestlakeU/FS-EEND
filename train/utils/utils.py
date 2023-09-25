import tqdm
import torch
from torch import Tensor
import numpy as np
import torch.nn.functional as F
from numpy import random as nr

from scipy.optimize import linear_sum_assignment

from torch.utils.data import WeightedRandomSampler

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