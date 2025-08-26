<div align="center">
<image src="/LS-EEND/utlis/arch.png"  width="300" alt="The proposed LS-EEND architecture" />
</div>

# Start training on simulated data by
python train_dia_simu.py --configs conf/spk_onl_conformer_retention_enc_dec_nonautoreg.yaml --gpus YOUR_DEVICE_ID,

# Finetune on real data by
python train_dia_fintun_real.py --configs spk_onl_conformer_retention_enc_dec_nonautoreg_callhome/ami/dihard2/dihard3.yaml --gpus YOUR_DEVICE_ID,

# Inference by (# modify your own path to save predictions in test_step in train/oln_tfm_enc_decxxx.py.)
python train_diaxxx.py --configs conf/xxx_infer.yaml --gpus YOUR_DEVICE_ID, --test_from_folder YOUR_CKPT_SAVE_DIR

# Evaluation
# generate speech activity probability (diarization results)
cd visualize
python gen_h5_output.py

# calculate DERs
python metrics.py --configs conf/xxx_infer.yaml

For simulated data and callhome data, we use a collar tolerance of 0.25s and median filtering, for ami, dihard2 and dihard3 data, no collar tolerance and no median filtering are used.

ami, dihard2 and dihard3 data are down-sampled to 8 kHz to align with the sampling rate of the simulated dataset.

# Performance

| Simulated Dataset | Simu1spk | Simu2spk | Simu3spk | Simu4spk | Simu5spk | Simu6spk | Simu7spk | Simu8spk |
| :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **DERS (%)** |  0.34 | 2.84 | 6.25 | 8.34 | 11.26 | 15.36 | 19.53 | 23.35 |
| **ckpt** | [simu_1-8spk.ckpt](https://drive.google.com/file/d/1uWY8JvjHJJ-SvGiNS-6s3q10g4CY2ePt/view?usp=sharing) | same | same | same | same | same | same | same |

|Real-world Dataset | CALLHOME | DIHARD II | DIHARD III | AMI Dev | AMI Eval |
| :--------: | :--: | :--: | :--: | :--: | :--: |
| **DERS (%)** | 12.11 | 27.58 | 19.61 | 20.97 | 20.76 |
| **ckpt** | [ch.ckpt](https://drive.google.com/file/d/1W8nYAB6YoEKMM5KZX-apVADvHaYc2Fre/view?usp=sharing) | [dih2.ckpt](https://drive.google.com/file/d/1vu7VSTnrNsooz5DzaodmctjdwblfB3wv/view?usp=sharing) | [dih3.ckpt](https://drive.google.com/file/d/115iaEG1OZwXa9tSyScXGtWeOk9JLfpER/view?usp=sharing) | [ami.ckpt](https://drive.google.com/file/d/1Zbc-8fXr_9kydjYS5SAeIaYDr6O1Ik74/view?usp=sharing) | [ami.ckpt](https://drive.google.com/file/d/1Zbc-8fXr_9kydjYS5SAeIaYDr6O1Ik74/view?usp=sharing) |


