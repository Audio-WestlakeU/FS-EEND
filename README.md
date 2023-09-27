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

# Note
Our codes are developed based on the original Chainer implementation of offline [EEND](https://github.com/hitachi-speech/EEND) by [hitachi Ltd.](https://github.com/hitachi-speech) and the Pytorch version [EEND-Pytorch](https://github.com/Xflick/EEND_PyTorch).
# Get start
1. Clone the FS-EEND codes by:

```
git clone https://github.com/Audio-WestlakeU/FS-EEND.git
```

2. Prepare kaldi-style data and modify conf/xxx.yaml according to your own paths.

3. Start training by

```
python train_dia.py --configs conf/spk_onl_tfm_enc_dec_nonautoreg.yaml --gpus YOUR_DEVICE_ID
```

4. Finetune on CALLHOME data by
