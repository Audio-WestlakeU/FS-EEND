import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--OMP_NUM_THREADS', type=int, default=1)
temp_args, _ = parser.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(temp_args.OMP_NUM_THREADS)


import torch
import pytorch_lightning as pl
from nnet.model.streaming_tfm_enc_1dcnn_enc_linear_non_autoreg_pos_enc_l2norm import StreamingTransformerEDADiarization
from nnet.model.onl_tfm_enc_1dcnn_enc_linear_non_autoreg_pos_enc_l2norm import OnlineTransformerDADiarization
from datasets.feature import * 
from train.utils.make_rttm import make_rttm
import hyperpyyaml
from nnet.utils.copy_params import copy_params_from_masked_to_streaming

import warnings
warnings.filterwarnings("ignore")


def predict(wav_path, configs, test_folder=None, test_file=None, gpus=0):
    # Set device
    # device = torch.device(f"cuda:{gpus}" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    # Extract Fbank feature
    feat = extract_fbank(
        wav_path,
        context_size=configs["data"]["context_recp"],
        input_transform=configs["data"]["feat_type"],
        frame_size=configs["data"]["feat"]["win_length"],
        frame_shift=configs["data"]["feat"]["hop_length"],
        subsampling=configs["data"]["subsampling"]
        ).to(device)
    rec = wav_path.split("/")[-1].split(".")[0]
    clip_len = feat.shape[0]

    # Define model
    masked_model = OnlineTransformerDADiarization(
        n_speakers=configs["data"]["num_speakers"],
        in_size=(2 * configs["data"]["context_recp"] + 1) * configs["data"]["feat"]["n_mels"], 
        **configs["model"]["params"],
    ).to(device)
    
    streaming_model = StreamingTransformerEDADiarization(
        in_size=(2 * configs["data"]["context_recp"] + 1) * configs["data"]["feat"]["n_mels"], 
        **configs["model"]["params"],
    ).to(device)
    
    # Load ckpt
    state_dict = torch.load(test_file, map_location="cpu")
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[len('model.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    # ckpt_package = torch.load(test_file, map_location="cpu")
    masked_model.load_state_dict(new_state_dict)
    masked_model.eval()
    masked_pred, _, _ = masked_model.test([feat], [len(feat)], max_nspks=configs["data"]["max_speakers"] + 2)
    masked_pred = torch.sigmoid(masked_pred[0][:, 1:])
    print(masked_pred)
    copy_params_from_masked_to_streaming(masked_model, streaming_model)

    # Predict
    preds = []
    streaming_model.eval()
    for t in range(len(feat)):
        feat_t = feat[t:t+1].unsqueeze(0) # (B, 1, D)
        pred_t = streaming_model.test(feat_t, max_nspks=configs["data"]["max_speakers"] + 2) # (B, 1, S)
        if pred_t is not None:
            preds.append(pred_t)
    for _ in range(configs["model"]["params"]["conv_delay"]):
        dummy_feat = torch.zeros(1, 1, feat.shape[-1], device=feat.device)
        pred_t = streaming_model.test(dummy_feat, max_nspks=configs["data"]["max_speakers"] + 2, dummy_conv_input=True) # (B, 1, S)
        if pred_t is not None:
            preds.append(pred_t)
    
    preds = torch.cat(preds, dim=1)
    pred = torch.sigmoid(preds[0][:, 1:])
    print(pred)
    # print(pred.shape)
    
    print(torch.allclose(pred, masked_pred, atol=1e-4, rtol=1e-4))
    # Dictionary with the key of speaker and the value of the predicted active timestamps of each speaker
    pred = pred.detach().cpu()
    rttm = make_rttm(rec=rec, 
              pred=pred, 
              frame_shift=configs["data"]["feat"]["hop_length"],
              subsampling=configs["data"]["subsampling"],
              sampling_rate=configs["data"]["feat"]["sample_rate"])
    return rttm

def plot_pred(pred):
    pass

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--wav_path", type=str, default="./test_samples/mix_0000176.wav", help="Path to input wav file")
    parser.add_argument('--configs', default='./conf/spk_onl_tfm_enc_dec_nonautoreg_infer.yaml', help='Configuration file path')
    parser.add_argument("--test_from_folder", default="./ckpt/simu", help="Checkpoint folder to test training")
    parser.add_argument("--test_from_file", default="./ckpt/simu/FS-EEND_simu_41_50epo_avg_model.ckpt", help="Checkpoint file to test training")
    parser.add_argument("--gpus", default=0, type=int, help="Device id of gpus to use")
    setup = parser.parse_args()
    with open(setup.configs, "r") as f:
        configs = hyperpyyaml.load_hyperpyyaml(f)
        f.close()

    rttm = predict(wav_path=setup.wav_path, configs=configs, test_folder=setup.test_from_folder, test_file=setup.test_from_file, gpus=setup.gpus)
    for spk in rttm:
        for utt in rttm[spk]:
            print(spk, utt)
