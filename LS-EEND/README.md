# repo: https://github.com/Audio-WestlakeU/FS-EEND
# contact: liangdi@westlake.edu.cn

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

# For simulated data and callhome data, we use a collar tolerance of 0.25s and median filtering, for ami, dihard2 and dihard3 data, no collar tolerance and no median filtering are used.

# ami, dihard2 and dihard3 data are down-sampled to 8 kHz to align with the sampling rate of the simulated dataset.
