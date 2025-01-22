import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--OMP_NUM_THREADS', type=int, default=1)
temp_args, _ = parser.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(temp_args.OMP_NUM_THREADS)


import torch
import pytorch_lightning as pl
from nnet.model.onl_tfm_enc_1dcnn_enc_linear_non_autoreg_pos_enc_l2norm import OnlineTransformerDADiarization
from datasets.feature import * 
from train.utils.make_rttm import make_rttm
from utlis.avg_ckpt import avg_ckpt
import hyperpyyaml
from train.oln_tfm_enc_dec import SpeakerDiarization

import warnings
warnings.filterwarnings("ignore")


def predict(wav_path, configs, test_folder=None):
    # Extract Fbank feature
    feat = extract_fbank(
        wav_path,
        context_size=configs["data"]["context_recp"],
        input_transform=configs["data"]["feat_type"],
        frame_size=configs["data"]["feat"]["win_length"],
        frame_shift=configs["data"]["feat"]["hop_length"],
        subsampling=configs["data"]["subsampling"]
        )
    rec = wav_path.split("/")[-1].split(".")[0]
    clip_len = feat.shape[0]

    # Define model
    model = OnlineTransformerDADiarization(
        n_speakers=configs["data"]["num_speakers"],
        in_size=(2 * configs["data"]["context_recp"] + 1) * configs["data"]["feat"]["n_mels"], 
        **configs["model"]["params"],
    )
    
    # Load ckpt
    state_dict = avg_ckpt(test_folder=test_folder)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[len('model.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)

    # Predict
    model.eval()
    preds, _, _ = model.test([feat], [clip_len], configs["data"]["max_speakers"] + 2)
    pred = torch.sigmoid(preds[0][:, 1:])
    # Dictionary with the key of speaker and the value of the predicted active timestamps of each speaker
    rttm = make_rttm(rec=rec, 
              pred=pred, 
              frame_shift=configs["data"]["feat"]["hop_length"],
              subsampling=configs["data"]["subsampling"],
              sampling_rate=configs["data"]["feat"]["sample_rate"])
    return rttm


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--wav_path", type=str, default="./swb_sre_ts_ns4_beta9_500/16/mix_0000077.wav", help="Path to input wav file")
    parser.add_argument('--configs', help='Configuration file path', default='./conf/spk_onl_tfm_enc_dec_nonautoreg_infer.yaml')
    parser.add_argument("--test_from_folder", default="./ckpt/simu", help="Checkpoint path to test training")
    setup = parser.parse_args()
    with open(setup.configs, "r") as f:
        configs = hyperpyyaml.load_hyperpyyaml(f)
        f.close()

    rttm = predict(wav_path=setup.wav_path, configs=configs, test_folder=setup.test_from_folder)
    for spk in rttm:
        for utt in rttm[spk]:
            print(spk, utt)
