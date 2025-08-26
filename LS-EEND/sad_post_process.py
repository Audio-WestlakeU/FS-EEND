"""
SAD post processing to filter false alarm and recover missed frames. 
Speech false alarm will be moved. Speech missed-frame errors will be replaced by correct estimation or at least speaker errors.

input:  y_hat: estimated postior probabilities of speech activity.
        z: oracle speech activity detection results.

output: y_hat_new: modified eatimation.

Copyright: Di Liang @ Audio Lab of Westlake University 2024
"""

import os
import numpy as np
import torch
from torch import Tensor
import hyperpyyaml
import h5py
from argparse import ArgumentParser
from datasets.diarization_dataset import KaldiDiarizationDataset


def sad_func(decision: Tensor, sad: Tensor, posterior: Tensor) -> Tensor:
    T, C = decision.shape
    decision_filter_fa = decision * sad
    # speech fa
    sph_fa = decision.sum() - decision_filter_fa.sum()
    zero_indices = (torch.sum(decision_filter_fa, dim=-1) == 0) & (sad.squeeze(dim=-1) == 1)
    max_indices = torch.argmax(posterior[zero_indices], dim=-1)
    decision_modified = decision_filter_fa.clone()
    decision_modified[zero_indices, max_indices] = 1
    
    return decision_modified


def sad_post_proposs(configs, hyp_dir, oup_dir, threshold=0.5, median=11, subsampling=10):
    test_set = KaldiDiarizationDataset(
            data_dir=configs["data"]["val_data_dir"],
            chunk_size=configs["data"]["chunk_size"],
            context_size=configs["data"]["context_recp"],
            input_transform=configs["data"]["feat_type"],
            frame_size=configs["data"]["feat"]["win_length"],
            frame_shift=configs["data"]["feat"]["hop_length"],
            subsampling=configs["data"]["subsampling"],
            rate=configs["data"]["feat"]["sample_rate"],
            label_delay=configs["data"]["label_delay"],
            n_speakers=configs["data"]["num_speakers"],
            use_last_samples=configs["data"]["use_last_samples"],
            shuffle=configs["data"]["shuffle"])
    subsampling = configs["data"]["subsampling"]
    der = 0.0
    diaerr = 0.0
    spkcon = 0.0
    falarm = 0.0
    miss = 0.0
    spkscore = 0.0
    for i in range(len(test_set)):
        # ref: (5000, C)
        label, rec = test_set.__getfulllabel__(i)

        # label = label[::subsampling]
        
        # convert label to sad
        sad, idx = label.max(dim=-1, keepdim=True)
        # read hypothesis h5 file
        filepath = os.path.join(hyp_dir, rec+".h5")
        data = h5py.File(filepath, 'r')
        posterior = torch.from_numpy(data['T_hat'][:]).float()
        decision = torch.where(posterior > threshold, 1, 0)
        modif_decision = sad_func(decision, sad, posterior)

        h5_file = h5py.File(oup_dir + f"/{rec}.h5", "w")
        h5_file.create_dataset("T_hat", data=modif_decision)
        h5_file.close()
        
        

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--configs', help='Configuration file path', required=True)
    parser.add_argument('--preds_dir', default=None, help='Hypothesis results dir')
    parser.add_argument("--thredshold", default=0.5, help="Threshold of decision")
    parser.add_argument("--median", default=11, help="Median filter parameter")
    setup = parser.parse_args()
    with open(setup.configs, "r") as f:
        configs = hyperpyyaml.load_hyperpyyaml(f)
        f.close()
    preds_dir = "./tsne_visual/data/onl_allspk_version_10w_ver_204_ami_ver_123_dev/preds_h5"
    oup_dir = "./tsne_visual/data/onl_allspk_version_10w_ver_204_ami_ver_123_dev/sad_post"

    if not os.path.isdir(oup_dir):
        os.mkdir(oup_dir)
    sad_post_proposs(configs, preds_dir, oup_dir)
    # T, S = 5, 3

    # y = torch.tensor([[0, 0, 1],
    #               [0, 0, 0],
    #               [0, 0, 0],
    #               [0, 1, 0],
    #               [0, 0, 0]], dtype=torch.float32)
    
    # z = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float32).unsqueeze(-1)

    # p = torch.rand(T, S)

    # print(p)

    # print(sad_func(y, z, p))

