# FS-EEND

The official Pytorch implementation of "[Frame-wise streaming end-to-end speaker diarization with non-autoregressive self-attention-based attractors](https://arxiv.org/abs/2309.13916)" [accepted by ICASSP 2024].

"[LS-EEND: Long-Form Streaming End-to-End Neural Diarization with Online Attractor Extraction](https://arxiv.org/abs/2410.06670)." [accepted by IEEE Trans. ASLPRO 2025].



<div>
    </p>
    <a href="https://github.com/Audio-WestlakeU/FS-EEND/"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/Audio-WestlakeU/FS-EEND/"><img src="https://img.shields.io/badge/Python-3.9-orange" alt="version"></a>
    <a href="https://github.com/Audio-WestlakeU/FS-EEND/"><img src="https://img.shields.io/badge/PyTorch-1.13-brightgreen" alt="python"></a>
    <a href="https://github.com/Audio-WestlakeU/FS-EEND/"><img src="https://img.shields.io/badge/PyTorchLightning-1.8-yellow" alt="python"></a>
</div>

[Paper :star_struck:](https://arxiv.org/abs/2309.13916) **|** [Issues :sweat_smile:](https://github.com/Audio-WestlakeU/FS-EEND/issues)
 **|** [Lab :hear_no_evil:](https://github.com/Audio-WestlakeU) **|** [Contact :kissing_heart:](liangdi@westlake.edu.cn)

# Introduction

This work proposes a frame-wise online/streaming end-to-end neural diarization (FS-EEND) method in a frame-in-frame-out fashion. To frame-wisely detect a flexible number of speakers and extract/update their corresponding attractors, we propose to leverage a causal speaker embedding encoder and an online non-autoregressive self-attention-based attractor decoder. A look-ahead mechanism is adopted to allow leveraging some future frames for effectively detecting new speakers in real time and adaptively updating speaker attractors.

<div align="center">
<image src="/FS-EEND/utlis/arch.png"  width="300" alt="The proposed FS-EEND architecture" />
</div>

# Get started
1. Clone the FS-EEND codes by:

```
git clone https://github.com/Audio-WestlakeU/FS-EEND.git
```

2. Prepare kaldi-style data by referring to [here](https://github.com/hitachi-speech/EEND/blob/master/egs/callhome/v1/run_prepare_shared.sh). Modify conf/xxx.yaml according to your own paths.

3. Start training on simulated data by

```
python train_dia.py --configs conf/spk_onl_tfm_enc_dec_nonautoreg.yaml --gpus YOUR_DEVICE_ID,
```

4. Modify your pretrained model path in conf/spk_onl_tfm_enc_dec_nonautoreg_callhome.yaml.
5. Finetune on CALLHOME data by
```
python train_dia_fintn_ch.py --configs conf/spk_onl_tfm_enc_dec_nonautoreg_callhome.yaml --gpus YOUR_DEVICE_ID,
```
6. Inference by (# modify your own path to save predictions in test_step in train/oln_tfm_enc_decxxx.py.)
```
python train_diaxxx.py --configs conf/xxx_infer.yaml --gpus YOUR_DEVICE_ID, --test_from_folder YOUR_CKPT_SAVE_DIR
```
7. Evaluation
 ```
# generate speech activity probability (diarization results)
cd visualize
python gen_h5_output.py

#calculate DERs
python metrics.py --configs conf/xxx_infer.yaml
```

# Performance
Please note we use Switchboard Cellular (Part 1 and 2) and 2005-2008 NIST Speaker Recognition Evaluation (SRE) to generate simulated data (including 4054 speakers).

| Dataset | DER(%) |ckpt|
| :--------: | :--: | :--: | 
| Simu1spk | 0.6 | [simu_avg_41_50epo.ckpt](https://drive.google.com/file/d/1JYr1zOxsHwQxIk9W4vwxzUfJFtaTQ02q/view?usp=sharing) |
| Simu2spk | 4.3 | same as above |
| Simu3spk | 9.8 | same as above |
| Simu4spk | 14.7 | same as above |
| CH2spk | 10.0 | [ch_avg_91_100epo.ckpt](https://drive.google.com/file/d/1i1Ow9IfPSwBRyRazY8-VX3z4ngDvSwx6/view?usp=sharing) |
| CH3spk | 15.3 | same as above |
| CH4spk | 21.8 | same as above |

The ckpts are the average of model parameters for the last 10 epochs.

If you want to check the performance of ckpt on CALLHOME:
```
python train_dia_fintn_ch.py --configs conf/spk_onl_tfm_enc_dec_nonautoreg_callhome_infer.yaml --gpus YOUR_DEVICE_ID, --test_from_folder YOUR_CKPT_SAVE_DIR
```
Note the modification of the code in train_dia_fintn_ch.py
```
ckpts = [x for x in all_files if (".ckpt" in x) and ("epoch" in x) and int(x.split("=")[1].split("-")[0])>=configs["log"]["start_epoch"] and int(x.split("=")[1].split("-")[0])<=configs["log"]["end_epoch"]]

state_dict = torch.load(test_folder + "/" + c, map_location="cpu")["state_dict"]
```
to
```
ckpts = [x for x in all_files if (".ckpt" in x)]

state_dict = torch.load(test_folder + "/" + c, map_location="cpu")
```

# Update
Upload our implementation of [EEND-EDA](https://arxiv.org/abs/2106.10654) and [EEND-EDA+FLEX-STB](https://arxiv.org/abs/2101.08473)
```
python train_offl_eend_eda.py --configs conf/spk_offl_eend_eda.yaml --gpus YOUR_DEVICE_ID,
python train_STB.py --configs conf/spk_STB.yaml --gpus YOUR_DEVICE_ID, --test_from_folder YOUR_CKPT_SAVE_DIR
```
Update predict code to support WAV input
```
python dia_pred.py --wav_path xxx.wav --configs conf/xxx_infer.yaml --test_from_folder YOUR_CKPT_SAVE_DIR
```

# Streaming Inference
```
python streaming_infer_dia.py
```

# Reference code
- <a href="https://github.com/hitachi-speech/EEND" target="_blank">EEND</a> 
- <a href="https://github.com/Xflick/EEND_PyTorch" target="_blank">EEND-Pytorch</a>

# Citation

If you want to cite this paper:

```
@INPROCEEDINGS{10446568,
  author={Liang, Di and Shao, Nian and Li, Xiaofei},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Frame-Wise Streaming end-to-end Speaker Diarization with Non-Autoregressive Self-Attention-Based Attractors}, 
  year={2024},
  pages={10521-10525}}
```
