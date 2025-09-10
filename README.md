# FS-EEND & LS-EEND

The official Pytorch implementation of:

[1] "[Frame-wise streaming end-to-end speaker diarization with non-autoregressive self-attention-based attractors](https://arxiv.org/abs/2309.13916)" [accepted by ICASSP 2024].

[2] "[LS-EEND: Long-Form Streaming End-to-End Neural Diarization with Online Attractor Extraction](https://ieeexplore.ieee.org/document/11122273)." [accepted by IEEE Trans. ASLPRO 2025].



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

This work proposes a frame-wise online/streaming end-to-end neural diarization (EEND) method, which detects speaker activities in a frame-in-frame-out fashion. The proposed model mainly consists of a causal embedding encoder and an online attractor decoder. Speakers are modeled in the self-attention-based decoder along both the time and speaker dimensions, and frame-wise speaker attractors are automatically generated and updated for new speakers and existing speakers, respectively. Retention mechanism is employed and especially adapted for long-form diarization with a linear temporal complexity. A multi-step progressive training strategy is proposed for gradually learning from easy tasks to hard tasks in terms of the number of speakers and audio length. Finally, the proposed model (referred to as long-form streaming EEND, LS-EEND) is able to perform streaming diarization for a high (up to 8) and flexible number speakers and very long (say one hour) audio recordings.

<div align="center">
<image src="/LS-EEND/utlis/arch.png"  width="600" alt="The proposed LS-EEND architecture" />
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
6. Inference by (# modify your own path to save predictions in test_step in train/oln_tfm_enc_decxxx.py. train_dia_simu.py for inferring simulated data and train_dia_fintun_real.py for inferring real-word data)
```
python train_dia_simu.py --configs conf/xxx_infer.yaml --gpus YOUR_DEVICE_ID, --test_from_folder YOUR_CKPT_SAVE_DIR
python train_dia_fintun_real.py --configs conf/xxx_infer.yaml --gpus YOUR_DEVICE_ID, --test_from_folder YOUR_CKPT_SAVE_DIR
```
7. Evaluation
 ```
# generate speech activity probability (diarization results)
cd visualize
python gen_h5_output.py

#calculate DERs (mid filter and collar)
python metrics.py --configs conf/xxx_infer.yaml
```

# Performance

| Simulated Dataset | Simu1spk | Simu2spk | Simu3spk | Simu4spk | Simu5spk | Simu6spk | Simu7spk | Simu8spk |
| :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **DERS (%)** |  0.34 | 2.84 | 6.25 | 8.34 | 11.26 | 15.36 | 19.53 | 23.35 |
| **ckpt** | [simu_1-8spk.ckpt](https://drive.google.com/file/d/1uWY8JvjHJJ-SvGiNS-6s3q10g4CY2ePt/view?usp=sharing) | same | same | same | same | same | same | same |

|Real-world Dataset | CALLHOME | DIHARD II | DIHARD III | AMI Dev | AMI Eval |
| :--------: | :--: | :--: | :--: | :--: | :--: |
| **DERS (%)** | 12.11 | 27.58 | 19.61 | 20.97 | 20.76 |
| **ckpt** | [ch.ckpt](https://drive.google.com/file/d/1W8nYAB6YoEKMM5KZX-apVADvHaYc2Fre/view?usp=sharing) | [dih2.ckpt](https://drive.google.com/file/d/1vu7VSTnrNsooz5DzaodmctjdwblfB3wv/view?usp=sharing) | [dih3.ckpt](https://drive.google.com/file/d/115iaEG1OZwXa9tSyScXGtWeOk9JLfpER/view?usp=sharing) | [ami.ckpt](https://drive.google.com/file/d/1Zbc-8fXr_9kydjYS5SAeIaYDr6O1Ik74/view?usp=sharing) | [ami.ckpt](https://drive.google.com/file/d/1Zbc-8fXr_9kydjYS5SAeIaYDr6O1Ik74/view?usp=sharing) |


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

@ARTICLE{11122273,
  author={Liang, Di and Li, Xiaofei},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={LS-EEND: Long-Form Streaming End-to-End Neural Diarization With Online Attractor Extraction}, 
  year={2025},
  volume={33},
  number={},
  pages={3568-3581},
  doi={10.1109/TASLPRO.2025.3597446}}
```
