cd ../../

gpus=$1
n_hidden=96
n_ffn=256
n_layers=16
clip_val=1e-10
log_eps_lseend=1e-10
lambda_spk=0.0001
FFT=200
N_FFT=256
HOP=80
label_look_ahead=9
model_look_ahead=0
ckpt_root="/mnt/home/liangdi/projects/pl_version/spa_enhan/spa/logs/CleanMel/16xSPB_Hid96_online_mrm/FLOPs_5.4G/VocosNorm_Clip1e-6/rts/version_4/checkpoints/"
model_name="weightavg40-49_40_49"
ckpt_path="${ckpt_root}/${model_name}"
lseend_ckpt_path="/data1/liangdi/logs/distant/spk_onl_tfm_enc_dec_10w_ami_and_noisy_and_clean/version_74/onlineConformerDA_cummn_retention_emb1dcnn_linear_nonautoreg_l2norm_pit_ami_and_noisy_and_clean_allspk_463_463_avg_model.ckpt"

python -m models.CleanMelTrainer_log10_mask_wi_emb_discrim_loss_simu fit \
    --config configs/models/cleanmel_and_lseend.yaml \
    --config configs/datasets/real_rir_rts_diar_label.yaml \
    --trainer.precision=32 \
    --trainer.num_sanity_val_steps=125 \
    --trainer.max_epochs=20 \
    --trainer.devices=${gpus} \
    --data.audio_time_len=[80,80,80] \
    --data.batch_size=[1,1] \
    --data.dataset_len=[20000,1000,1000] \
    --data.target='rts_0.15' \
    --model.exp_name="/fronted_wi_emb_consis_loss_simu/${n_layers}xSPB_Hid${n_hidden}_online_mrm/FLOPs_5.4G/VocosNorm_Clip${clip_val}/rts/" \
    --model.log_eps=${clip_val} \
    --model.arch.init_args.num_layers=${n_layers} \
    --model.arch.init_args.dim_hidden=${n_hidden} \
    --model.arch.init_args.dim_input=2 \
    --model.input_stft.init_args.n_win=${FFT} \
    --model.input_stft.init_args.n_fft=${N_FFT} \
    --model.input_stft.init_args.n_hop=${HOP} \
    --model.input_stft.init_args.online=true \
    --model.target_stft.init_args.n_win=${FFT} \
    --model.target_stft.init_args.n_fft=${N_FFT} \
    --model.target_stft.init_args.n_hop=${HOP}  \
    --model.target_stft.init_args.power=2 \
    --model.target_stft.init_args.mel_norm="slaney" \
    --model.target_stft.init_args.mel_scale="slaney" \
    --model.target_stft.init_args.librosa_mel=True \
    --model.target_stft.init_args.online=true \
    --model.arch.init_args.online=true \
    --model.arch.init_args.encoder_kernel_size=5 \
    --model.arch.init_args.look_ahead=${model_look_ahead} \
    --model.metrics='[]' \
    --model.vocos_config=null \
    --model.vocos_ckpt=null \
    --model.look_ahead=${label_look_ahead} \
    --model.ckpt_path=${ckpt_path} \
    --model.lseend_ckpt_path=${lseend_ckpt_path} \
    --model.log_eps_lseend=${log_eps_lseend} \
    --model.lambda_spk=${lambda_spk} \
    # --ckpt_path='/mnt/home/liangdi/projects/pl_version/spa_enhan/spa/logs/CleanMel/16xSPB_Hid96_online_mrm/FLOPs_5.4G/VocosNorm_Clip1e-6/rts/version_51/checkpoints/last.ckpt' \
    # --model.ckpt_path=${ckpt_path} \
    # --model.ckpt_path=${ckpt_path} \
    # --model.arch_ckpt=${ckpt_path} \