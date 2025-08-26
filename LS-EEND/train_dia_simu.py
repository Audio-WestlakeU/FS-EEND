##############################################################################################################
# This file is used to train the Online Conformer Retention Diarization model with encoder-decoder structure.
#
# Copyright: Di Liang @ Audio Lab of Westlake University 2024
##############################################################################################################

import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--OMP_NUM_THREADS', type=int, default=1)
temp_args, _ = parser.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(temp_args.OMP_NUM_THREADS)

import random
import torch
# torch.set_num_threads(1)
import yaml
import hyperpyyaml

import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.profilers import AdvancedProfiler

from functools import partial
from collections import defaultdict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from nnet.model.onl_conformer_retention_enc_1dcnn_tfm_retention_enc_linear_non_autoreg_pos_enc_l2norm_emb_loss_mask import OnlineConformerRetentionDADiarization
from utlis.scheduler import NoamScheduler
from datasets.diarization_dataset_on_the_fly import KaldiDiarizationDataset, my_collate
from train.oln_tfm_enc_dec_on_the_fly import SpeakerDiarization

import time
# from torch.utils.flop_counter import FlopCounterMode # requires torch>=2.1.0

# from torchprofile import profile_macs

import warnings
warnings.filterwarnings("ignore")


def train(configs, gpus, checkpoint_resume, test_folder=None):
    train_set = KaldiDiarizationDataset(
            data_dir=configs["data"]["train_data_dir"],
            data_type="train",
            chunk_size=configs["data"]["chunk_size"],
            chunk_step=configs["data"]["chunk_step"],
            context_size=configs["data"]["context_recp"],
            input_transform=configs["data"]["feat_type"],
            frame_size=configs["data"]["feat"]["win_length"],
            frame_shift=configs["data"]["feat"]["hop_length"],
            subsampling=configs["data"]["subsampling"],
            rate=configs["data"]["feat"]["sample_rate"],
            label_delay=configs["data"]["label_delay"],
            n_speakers=configs["data"]["num_speakers"],
            use_last_samples=configs["data"]["use_last_samples"],
            shuffle=configs["data"]["shuffle"])

    val_set = KaldiDiarizationDataset(
            data_dir=configs["data"]["val_data_dir"],
            data_type = "val",
            chunk_size=configs["data"]["val_chunk_size"],
            chunk_step=configs["data"]["val_chunk_step"],
            context_size=configs["data"]["context_recp"],
            input_transform=configs["data"]["feat_type"],
            frame_size=configs["data"]["feat"]["win_length"],
            frame_shift=configs["data"]["feat"]["hop_length"],
            subsampling=configs["data"]["subsampling"],
            rate=configs["data"]["feat"]["sample_rate"],
            label_delay=configs["data"]["label_delay"],
            n_speakers=configs["data"]["num_speakers"],
            use_last_samples=configs["data"]["use_last_samples"],
            shuffle=configs["data"]["shuffle"])
    
    datasets = {
        "train": train_set,
        "val": val_set
    }
    
    collate_func = my_collate

    # Define model
    model = OnlineConformerRetentionDADiarization(
        n_speakers=configs["data"]["num_speakers"],
        in_size=(2 * configs["data"]["context_recp"] + 1) * configs["data"]["feat"]["n_mels"],          # Transformer need to know maximum data length
        # max_seqlen=chunk_size,
        **configs["model"]["params"],
    )


    # Define optimizer
    opt_config = {
        "params": model.parameters(),
        "lr": configs["training"]["lr"]
    }
    opt_name = configs["training"]["opt"].lower()
    if opt_name == "adam":
        opt = torch.optim.Adam
    elif opt_name == "sgd":
        opt = torch.optim.SGD
    elif opt_name == "noam":
        opt = partial(
            torch.optim.Adam,
            betas=(0.9, 0.98),
            eps=1e-9
        )
    else: 
        NotImplementedError
    opt = opt(**opt_config)

    if configs["training"]["scheduler"]:
        print("Using noam scheduler")
        scheduler = NoamScheduler(opt, configs["model"]["params"]["n_units"], configs["training"]["warm_steps"], scale=configs["training"]["schedule_scale"]) if configs["training"]["scheduler"].lower() == "noam" else NotImplementedError
    else:
        scheduler = None

    # Define the logger
    logger = TensorBoardLogger(os.path.dirname(configs["log"]["log_dir"]), configs["log"]["model_name"])
    configs["log"]["log_dir"] = logger.log_dir  # updarte log_dir to: ./logs/model_xx/version_xx
    print("Experiment dir:", configs["log"]["log_dir"])
    os.makedirs(configs["log"]["log_dir"], exist_ok=True)
    with open(configs["log"]["log_dir"] + "/config.yaml", "w") as f:
        docs = yaml.dump(configs, f)
        f.close()
    callbacks = [
        EarlyStopping(monitor="val/obj_metric", patience=configs["training"]["early_stop_epoch"], verbose=True, mode="min"),
        ModelCheckpoint(logger.log_dir, monitor="val/obj_metric", save_top_k=configs["log"]["save_top_k"], mode="min", save_last=True)
        # ModelCheckpoint(logger.log_dir, monitor="val/obj_metric", save_top_k=configs["log"]["save_top_k"], mode="min", save_last=True, every_n_train_steps=configs["training"]["every_n_train_steps"])
    ]


    # Define the training setup
    spk_dia_main = SpeakerDiarization(
        hparams=configs,
        model=model,
        datasets=datasets,
        opt=opt,
        scheduler=scheduler,
        collate_func=collate_func
    )

    # Initialization
    if configs["training"]["init_ckpt"]:
        print("Load from checkpoint {} ... ".format(configs["training"]["init_ckpt"]))
        ckpt_package = torch.load(configs["training"]["init_ckpt"], map_location="cpu")
        # state_dict = ckpt_package["state_dict"]
        spk_dia_main.load_state_dict(ckpt_package)
    
    # Freezing parameters
    # Freeze all
    # for name, params in model.named_parameters():
    #     if "dec" in name:
    #         print("Unfreeze:", name)
    #         params.requires_grad = True
    #     else:
    #         print("Freeze:", name)
    #         params.requires_grad = False

    # Define the trainer
    profiler = AdvancedProfiler(filename="perf_logs")
    trainer = pl.Trainer(
        max_epochs=configs["training"]["max_epochs"],
        callbacks=callbacks,
        gpus=gpus,
        strategy=configs["training"]["dist_strategy"],
        sync_batchnorm=configs["training"]["sync_batchnorm"],
        accumulate_grad_batches=configs["training"]["grad_accm"],
        logger=logger,
        resume_from_checkpoint=checkpoint_resume,
        gradient_clip_val=configs["training"]["grad_clip"],
        check_val_every_n_epoch=configs["training"]["val_interval"],
        # val_check_interval=configs["training"]["every_n_train_steps"],
        **configs["debug"]
    )

    if test_folder is None:
        # Start training
        trainer.fit(spk_dia_main)
        best_path = trainer.checkpoint_callback.best_model_path
        print("Best model path:", best_path)
        test_folder = os.path.dirname(best_path)
    
    # Test step: given the folder and average the models in the given folder
    for _, _, files in os.walk(test_folder):
        all_files = files
    ckpts = [x for x in all_files if (".ckpt" in x)]
    # ckpts = [x for x in all_files if (".ckpt" in x) and ("epoch" in x) and int(x.split("=")[1].split("-")[0])>=configs["log"]["start_epoch"] and int(x.split("=")[1].split("-")[0])<=configs["log"]["end_epoch"]]

    print("Test using ckpts:")
    [print(test_folder + "/" + x) for x in ckpts]
    test_state = defaultdict(float)
    for c in ckpts:
        # state_dict = torch.load(test_folder + "/" + c, map_location=torch.device("cuda:{}".format(gpus[0])))["state_dict"]
        # state_dict = torch.load(test_folder + "/" + c)["state_dict"]
        # if only model parameters are saved, load state dict directly
        state_dict = torch.load(test_folder + "/" + c)
        for name, param in state_dict.items():
            test_state[name] += param / len(ckpts)

    if configs["log"]["save_avg_path"]:
        torch.save(test_state, configs["log"]["save_avg_path"])
    spk_dia_main.load_state_dict(test_state)
    trainer.test(spk_dia_main)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--configs', help='Configuration file path', required=True)
    parser.add_argument('--gpus', default=None, help='Device used for training')
    parser.add_argument("--checkpoint_resume", default=None, help="Checkpoint path to resume training")
    parser.add_argument("--test_from_folder", default=None, help="Checkpoint path to test training")
    parser.add_argument("--chunk_size", default=None, help="chunk_size")
    parser.add_argument("--der_output_file", default=None, help="der_output_file")
    # parser.add_argument("--start_epo", default=None, help="start_epo")
    # parser.add_argument("--end_epo", default=None, help="end_epo")
    setup = parser.parse_args()
    with open(setup.configs, "r") as f:
        configs = hyperpyyaml.load_hyperpyyaml(f)
        f.close()

    # Freeze seed
    seed = configs["training"]["seed"]
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    train(configs, gpus=setup.gpus, checkpoint_resume=setup.checkpoint_resume, test_folder=setup.test_from_folder)

    
