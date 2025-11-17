import os
import torch
import yaml
import hyperpyyaml
import torch.nn.functional as F
import h5py

from scipy.signal import medfilt
from argparse import ArgumentParser
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationErrorRate
from datasets.diarization_dataset import KaldiDiarizationDataset


def gen_ref(configs, hyp_dir, metric, threshold=0.5, median=11, subsampling=10):
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

        # perm label
        T, n_spk = label.shape
        frame_idx = torch.arange(1, T + 1).unsqueeze(-1)
        label_idx = (frame_idx * label)
        label_idx = label_idx.masked_fill_(label_idx == 0, torch.inf)
        sort_idx = torch.argsort(torch.min(label_idx, dim=0)[0])
        label = label[:, sort_idx]

        reference = Annotation(uri=f'file{i+1}')
        
        for spkid, frames in enumerate(label.T):
            frames = F.pad(frames, (1, 1), 'constant')
            changes, = torch.where(torch.diff(frames, dim=0) != 0)
            for s, e in zip(changes[::2], changes[1::2]):
                reference[Segment(s, e)] = str(spkid)
        
        # read hypothesis h5 file
        filepath = os.path.join(hyp_dir, rec+".h5")
        data = h5py.File(filepath, 'r')
        pred = torch.where(torch.from_numpy(data['T_hat'][:]).float() > threshold, 1, 0)
        if median > 1:
            pred = medfilt(pred, (median, 1))
        
            pred = torch.from_numpy(pred).float()
        hypothesis =  Annotation(uri=f'file{i+1}')
        for spkid, frames in enumerate(pred.T):
            frames = F.pad(frames, (1, 1), 'constant')
            changes, = torch.where(torch.diff(frames, dim=0) != 0)
            for s, e in zip(changes[::2], changes[1::2]):
                hypothesis[Segment(s*subsampling, e*subsampling)] = str(spkid)
        
        res = metric(reference, hypothesis, detailed=True)
        spkscore += res['total']
        spkcon += res['confusion']
        falarm += res['false alarm']
        miss += res['missed detection']
        diaerr += res['confusion'] + res['false alarm'] + res['missed detection']
        print(rec)
        print("der: ", (res['confusion'] + res['false alarm'] + res['missed detection']) / res['total'])

    der = diaerr / spkscore
    
    print("speaker score: ", spkscore)
    print('mean der: ', der)
    print('mean speaker confusion rate: ', spkcon / spkscore)
    print('mean speaker false alarm rate: ', falarm / spkscore)
    print('mean speaker miss rate: ', miss / spkscore)

            

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
    datatype = "preds_h5"
    # datatype = "sad_post"
    # datatype = "self_sad_post"
    preds_dir = f"/mnt/home/liangdi/projects/pl_version/ls_eend/outputs/data/onl_allspk_version_10w_ver_204_ch_ver_88/{datatype}"

    metric = DiarizationErrorRate(collar=50)  # Tolerance of 50 frames on both ends = 25 frames tolerance = 250ms tolerance
    gen_ref(configs, preds_dir, metric)
    print(datatype)
    print(preds_dir)
