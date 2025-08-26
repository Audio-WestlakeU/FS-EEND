

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

| Dataset | DER(%) |ckpt|
| :--------: | :--: | :--: | 
| Simu1spk | 0.34 | [simu_1-8spk.ckpt](https://drive.google.com/file/d/1uWY8JvjHJJ-SvGiNS-6s3q10g4CY2ePt/view?usp=sharing) |
| Simu2spk | 2.84 | same as above |
| Simu3spk | 6.25 | same as above |
| Simu4spk | 8.34 | same as above |
| Simu5spk | 11.26 | same as above |
| Simu6spk | 15.36 | same as above |
| Simu7spk | 19.53 | same as above |
| Simu8spk | 23.35 | same as above |
| CALLHOME | 12.11 | [ch.ckpt](https://drive.google.com/file/d/1W8nYAB6YoEKMM5KZX-apVADvHaYc2Fre/view?usp=sharing) |
| DIHARD II | 27.58 | [dih2.ckpt](https://drive.google.com/file/d/1vu7VSTnrNsooz5DzaodmctjdwblfB3wv/view?usp=sharing) |
| DIHARD III | 19.61 | [dih3.ckpt](https://drive.google.com/file/d/115iaEG1OZwXa9tSyScXGtWeOk9JLfpER/view?usp=sharing) |
| AMI Dev | 20.97 | [ami.ckpt](https://drive.google.com/file/d/1Zbc-8fXr_9kydjYS5SAeIaYDr6O1Ik74/view?usp=sharing) |
| AMI Eval | 20.76 | same as above |
