import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--OMP_NUM_THREADS', type=int, default=1)
temp_args, _ = parser.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(temp_args.OMP_NUM_THREADS)

import time
import warnings
import torch
import hyperpyyaml

from nnet.model.onl_conformer_retention_enc_1dcnn_tfm_retention_enc_linear_non_autoreg_pos_enc_l2norm_emb_loss_mask import (
    OnlineConformerRetentionDADiarization,
    StreamingConv1d,
)
from datasets.feature import extract_fbank
from train.utils.make_rttm import make_rttm

warnings.filterwarnings("ignore")


def build_streaming_cnn(model: OnlineConformerRetentionDADiarization, device):
    """Build a StreamingConv1d and copy weights from model.cnn.

    Kept outside the model so its params are NOT part of state_dict
    (original ckpt has no streaming_cnn.* keys).
    """
    kernel_size = 2 * model.delay + 1
    streaming_cnn = StreamingConv1d(model.n_units, model.n_units, kernel_size=kernel_size).to(device)
    streaming_cnn.conv.load_state_dict(model.cnn.state_dict())
    streaming_cnn.eval()
    return streaming_cnn


def init_streaming_states(model: OnlineConformerRetentionDADiarization, batch_size: int, device):
    n_enc_layers = len(model.enc.encoder.layers)
    n_dec_layers = len(model.dec.layers)
    enc_conv_ksize = model.enc.encoder._conv_kernel_size
    enc_states = {
        'ret_states': [dict() for _ in range(n_enc_layers)],
        'conv_caches': [
            torch.zeros(batch_size, model.n_units, enc_conv_ksize - 1, device=device)
            for _ in range(n_enc_layers)
        ],
    }
    dec_states = [dict() for _ in range(n_dec_layers)]
    return enc_states, dec_states


def streaming_predict(model, streaming_cnn, feat, max_nspks, device):
    """Run frame-by-frame streaming inference.

    feat: (T, D_in) feature tensor on `device`.
    Returns: (T, max_nspks) logit tensor.
    """
    B = 1
    T = feat.shape[0]
    enc_states, dec_states = init_streaming_states(model, B, device)

    # Reset streaming_cnn internal state each call
    streaming_cnn.buffer.clear()
    streaming_cnn.t = 0

    preds = []
    dec_t = 0

    def step(emb_t, dec_t_local):
        emb_t_conv = streaming_cnn(emb_t.transpose(1, 2))
        if emb_t_conv is None:
            return None, dec_t_local
        emb_t_conv = emb_t_conv.transpose(1, 2)
        emb_t_conv = emb_t_conv / torch.norm(emb_t_conv, dim=-1, keepdim=True)
        attractor_t = model.dec.forward_one_step(emb_t_conv, dec_t_local, max_nspks, dec_states)
        attractor_t = attractor_t / torch.norm(attractor_t, dim=-1, keepdim=True)
        y_t = torch.matmul(emb_t_conv.unsqueeze(dim=-2), attractor_t.transpose(-1, -2)).squeeze(dim=-2)
        return y_t, dec_t_local + 1

    # Main loop: real input frames
    for t in range(T):
        x_t = feat[t:t + 1].unsqueeze(0)  # (1, 1, D_in)
        emb_t = model.enc.forward_one_step(
            x_t, t, enc_states['ret_states'], enc_states['conv_caches']
        )
        y_t, dec_t = step(emb_t, dec_t)
        if y_t is not None:
            preds.append(y_t)

    # Flush: feed `conv_delay` zero frames so StreamingConv1d emits the final real-frame outputs
    for _ in range(model.delay):
        emb_zero = torch.zeros(B, 1, model.n_units, device=device)
        y_t, dec_t = step(emb_zero, dec_t)
        if y_t is not None:
            preds.append(y_t)

    return torch.cat(preds, dim=1).squeeze(0)  # (T, max_nspks)


def predict(wav_path, configs, test_file, gpus=0, output_rttm=None, compare_with_batch=True):
    device = torch.device(f"cuda:{gpus}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    feat = extract_fbank(
        wav_path,
        context_size=configs["data"]["context_recp"],
        input_transform=configs["data"]["feat_type"],
        frame_size=configs["data"]["feat"]["win_length"],
        frame_shift=configs["data"]["feat"]["hop_length"],
        subsampling=configs["data"]["subsampling"],
    ).to(device)
    rec = os.path.splitext(os.path.basename(wav_path))[0]
    T = feat.shape[0]
    print(f"[INFO] {T} frames extracted from {wav_path}")

    in_size = (2 * configs["data"]["context_recp"] + 1) * configs["data"]["feat"]["n_mels"]
    model = OnlineConformerRetentionDADiarization(
        n_speakers=configs["data"]["num_speakers"],
        in_size=in_size,
        **configs["model"]["params"],
    ).to(device)

    # Load ckpt; strip lightning 'model.' prefix if present
    state_dict = torch.load(test_file, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state_dict.items()}
    # ckpt was saved with old key 'dec.attractor_decoder.layers.*'; remap to 'dec.layers.*'
    cleaned = {k.replace("dec.attractor_decoder.layers.", "dec.layers."): v for k, v in cleaned.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[WARN] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.eval()

    streaming_cnn = build_streaming_cnn(model, device)
    max_nspks = configs["data"]["max_speakers"] + 2

    batch_pred = None
    if compare_with_batch:
        with torch.no_grad():
            batch_out, _, _ = model.test([feat], [len(feat)], max_nspks=max_nspks)
            batch_pred = torch.sigmoid(batch_out[0][:, 1:]).detach().cpu().float()
        print(f"[INFO] batch pred shape: {tuple(batch_pred.shape)}")

    st = time.time()
    with torch.no_grad():
        preds = streaming_predict(model, streaming_cnn, feat, max_nspks, device)
    ed = time.time()
    print(f"[INFO] streaming time: {ed - st:.2f}s total, "
          f"{(ed - st) * 1000 / max(T, 1):.2f} ms/frame")

    stream_pred = torch.sigmoid(preds[:, 1:]).detach().cpu().float()
    print(f"[INFO] streaming pred shape: {tuple(stream_pred.shape)}")

    if batch_pred is not None and batch_pred.shape == stream_pred.shape:
        max_diff = (batch_pred - stream_pred).abs().max().item()
        ok = torch.allclose(batch_pred, stream_pred, atol=1e-3, rtol=1e-3)
        print(f"[INFO] batch vs streaming: match={ok}, max diff={max_diff:.2e}")

    rttm = make_rttm(
        rec=rec,
        pred=stream_pred,
        frame_shift=configs["data"]["feat"]["hop_length"],
        subsampling=configs["data"]["subsampling"],
        sampling_rate=configs["data"]["feat"]["sample_rate"],
    )
    if output_rttm is not None:
        os.makedirs(os.path.dirname(output_rttm) or ".", exist_ok=True)
        with open(output_rttm, "w") as f:
            for spk in rttm:
                for line in rttm[spk]:
                    f.write(line + "\n")
        print(f"[INFO] RTTM saved to {output_rttm}")

    return rttm


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wav_path", type=str, default="./test_samples/mix_0000176.wav")
    parser.add_argument("--configs", type=str,
                        default="./conf/spk_onl_conformer_retention_enc_dec_nonautoreg_infer.yaml")
    parser.add_argument("--test_from_file", type=str,
                        default="./ckpt/simu/ls_eend_1-8spk_16_25_avg_model.ckpt")
    parser.add_argument("--output_rttm", type=str, default="./test_samples/streaming_predicted.rttm")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--no_compare", action="store_true",
                        help="skip batch inference sanity check")
    setup = parser.parse_args()

    with open(setup.configs, "r") as f:
        configs = hyperpyyaml.load_hyperpyyaml(f)

    rttm = predict(
        wav_path=setup.wav_path,
        configs=configs,
        test_file=setup.test_from_file,
        gpus=setup.gpus,
        output_rttm=setup.output_rttm,
        compare_with_batch=not setup.no_compare,
    )
    for spk in rttm:
        for line in rttm[spk]:
            print(spk, line)
