# FS-EEND

The official Pytorch implementation of "Frame-wise streaming end-to-end speaker diarization with non-autoregressive self-attention-based attractors".

This work is submitted to ICASSP 2024.

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
<image src="/utlis/arch.png"  width="300" alt="The proposed FS-EEND architecture" />
</div>

# Get started
1. Clone the FS-EEND codes by:

```
git clone https://github.com/Audio-WestlakeU/FS-EEND.git
```

2. Prepare kaldi-style data by referring to [here](https://github.com/hitachi-speech/EEND/blob/master/egs/callhome/v1/run_prepare_shared.sh). Modify conf/xxx.yaml according to your own paths.

3. Start training on simulated data by

```
python train_dia.py --configs conf/spk_onl_tfm_enc_dec_nonautoreg.yaml --gpus YOUR_DEVICE_ID
```

4. Modify your pretrained model path in conf/spk_onl_tfm_enc_dec_nonautoreg_callhome.yaml.
5. Finetune on CALLHOME data by
```
python train_dia_fintn.py --configs conf/spk_onl_tfm_enc_dec_nonautoreg_callhome.yaml --gpus YOUR_DEVICE_ID
```
6. Inference by
```
python train_diaxxx.py --configs conf/xxx_infer.yaml --gpus YOUR_DEVICE_ID --test_from_folder YOUR_CKPT_SAVE_DIR
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
| Simu1spk | 0.6 |
| Simu2spk | 5.1 |
| Simu3spk | 11.1 |
| Simu4spk | 15.8 |
| CH2spk | 10.1 |
| CH3spk | 14.6 |
| CH4spk | 21.2 |


# Reference code
- <a href="https://github.com/hitachi-speech/EEND" target="_blank">EEND</a> 
- <a href="https://github.com/Xflick/EEND_PyTorch" target="_blank">EEND-Pytorch</a> 
