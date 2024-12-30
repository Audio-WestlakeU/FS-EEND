import os
import torch

import numpy as np
import pytorch_lightning as pl

from collections import defaultdict
from pprint import pprint
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from .utils.loss import batch_pit_n_speaker_loss, pit_loss_multispk, report_diarization_error, standard_loss, pad_labels, pad_preds
from .utils.utils import ContextBuilder, TorchScaler

import time

class SpeakerDiarization(pl.LightningModule):
    """
    Speaker diarization main stream: 
    TODO: 
        test step is currently the same as the validation step
    """
    def __init__(self, hparams, model, datasets, opt, scheduler, collate_func) -> None:
        super(SpeakerDiarization, self).__init__()
        self.hparams.update(hparams)
        # Training configs
        self.datasets = datasets
        self.model = model
        self.opt = opt
        self.collate_func = collate_func
        self.scheduler = scheduler
        # self.loss_func = PITLoss(self.hparams["data"]["num_speakers"])
        self.loss_func1 = batch_pit_n_speaker_loss
        self.loss_func2 = standard_loss
        self.pad_preds = pad_preds
        self.pad_labels = pad_labels


    def scaler_init(self):
        return TorchScaler(
                self.hparams["data"]["scaler"]["statistic"],
                self.hparams["data"]["scaler"]["normtype"],
                self.hparams["data"]["scaler"]["dims"],
            )

    
    def detect(self, x, labels, ilens):
        # Here we could apply normalization or something else
        return self.model(x, tgt=labels, ilens=ilens)
    
    def training_step(self, batch, batch_index):
        feats, labels, _ = batch
        n_spks = [l.shape[1] for l in labels]
        clip_lengths = [x.shape[0] for x in feats]
        # Transform feats to MelSpectrum
        preds, attr_loss, _, _ = self.detect(feats, labels, clip_lengths)
        # print(preds.shape)

        max_nspk = max(n_spks)
        preds_pad = self.pad_preds(preds, max_nspk)
        labels = list(labels)
        labels_pad = self.pad_labels(labels, max_nspk)

        pit_loss1, perm_labels = self.loss_func1(preds_pad, labels_pad, n_spks)
        pit_loss = self.loss_func2(preds, perm_labels)

        # Calculate the total loss
        tot_loss = pit_loss + attr_loss
        self.log("train/lr", self.opt.param_groups[-1]["lr"], prog_bar=True, sync_dist=True)
        self.log("train/pit_loss", pit_loss, sync_dist=True)
        self.log("train/attr_loss", attr_loss, sync_dist=True)
        self.log("train/tot_loss", tot_loss,sync_dist=True)

        return tot_loss

    def validation_step(self, batch, batch_index):
        feats, labels, _ = batch
        n_spks = [l.shape[1] for l in labels]
        # Model pred
        clip_lengths = [x.shape[0] for x in feats]

        preds, attr_loss, _, _ = self.detect(feats, labels, clip_lengths)
        
        max_nspk = max(n_spks)
        preds_pad = self.pad_preds(preds, max_nspk)
        labels = list(labels)
        labels_pad = self.pad_labels(labels, max_nspk)
        
        pit_loss1, perm_labels = self.loss_func1(preds_pad, labels_pad, n_spks)
        pit_loss = self.loss_func2(preds, perm_labels)
        # print(f"batch_lost_n_speakers: {pit_loss1},  batch_pit_loss: {pit_loss3},  pit_loss: {pit_loss}")
        tot_loss = pit_loss + attr_loss

        # Metrics
        stats = report_diarization_error(preds, perm_labels)

        self.log("val/pit_loss", pit_loss, sync_dist=True)
        self.log("val/attr_loss", attr_loss, sync_dist=True)
        self.log("val/tot_loss", tot_loss, sync_dist=True)

        return stats
    
    def validation_epoch_end(self, val_step_outputs) -> None:
        stats_holder = defaultdict(list)
        for stats in val_step_outputs:
            for k, v in stats.items():
                stats_holder[k] += v
        # Calculate DER
        stats_avg = {k: sum(v)/len(v) for k, v in stats_holder.items()}
        stats_avg["DER"] = stats_avg["diarization_error"] / stats_avg["speaker_scored"]
        obj_metric = stats_avg['DER']
        self.log("val/obj_metric", obj_metric, sync_dist=True)
        self.log("val/DER", stats_avg['DER'], sync_dist=True)
        self.log("val/speech_scored", stats_avg['speech_scored'], sync_dist=True)
        self.log("val/speech_miss", stats_avg['speech_miss'], sync_dist=True)
        self.log("val/speech_falarm", stats_avg['speech_falarm'], sync_dist=True)
        self.log("val/speaker_scored", stats_avg['speaker_scored'], sync_dist=True)
        self.log("val/speaker_miss", stats_avg['speaker_miss'], sync_dist=True)
        self.log("val/speaker_falarm", stats_avg['speaker_falarm'], sync_dist=True)
        self.log("val/speaker_error", stats_avg['speaker_error'], sync_dist=True)
        self.log("val/diarization_error", stats_avg['diarization_error'], sync_dist=True)
        self.log("val/frames", stats_avg['frames'], sync_dist=True)


    def test_step(self, batch, batch_index):
        feats, labels, rec = batch
        n_spks = [l.shape[1] for l in labels]
        # Model pred
        clip_lengths = [x.shape[0] for x in feats]

        # preds, attractors_loss, embs, attractors = self.detect(feats, labels, clip_lengths)
        preds, embs, attractors = self.model.test(feats, clip_lengths, th=0.5)
        # pit_loss, perm_labels = self.loss_func1(preds, labels)
        # pit_loss = self.loss_func2(preds, perm_labels)
        # tot_loss = pit_loss + attr_loss

        # Metrics
        max_nspk = max(preds[0].shape[1], labels[0].shape[1])
        labels = list(labels)
        labels = pad_labels(labels, max_nspk)
        preds = pad_preds(preds, max_nspk)
        # preds_real = [pred[:, :nspk] for pred, nspk in zip(preds, n_spks)]
        pit_loss, perm_labels = self.loss_func1(preds, labels, [max_nspk])
        stats = report_diarization_error(preds, perm_labels)
        # stats = report_diarization_error(labels, labels)
        # if labels[0].shape[1] > 2:
        #     print(rec)
        num_spks = "1"
        version = "_off_ver_35"
        save_dir_parnt = f"./data/offl_{num_spks}spk_version{version}"
        for content in ["preds", "labels", "embs", "attractors"]:
            save_dir = os.path.join(save_dir_parnt, content)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        # np.save(f"./data/offl_{num_spks}spk_version{version}/preds/{rec[0]}.npy", preds[0].detach().cpu().numpy())
        # np.save(f"./data/offl_{num_spks}spk_version{version}/labels/label_{rec[0]}.npy", labels[0].detach().cpu().numpy())
        # np.save(f"./data/offl_{num_spks}spk_version{version}/embs/emb_{rec[0]}.npy", embs[0].detach().cpu().numpy())
        # np.save(f"./data/offl_{num_spks}spk_version{version}/attractors/attractor_{rec[0]}.npy", attractors[0].detach().cpu().numpy())
        return stats

    def test_epoch_end(self, test_step_outputs) -> None:
        stats_holder = defaultdict(list)
        for stats in test_step_outputs:
            for k, v in stats.items():
                stats_holder[k] += v
        # [print(x) for x in stats_holder["speaker_scored"] if not x ]
        # Calculate DER
        
        stats_avg = {k: sum(v)/len(v) for k, v in stats_holder.items()}
        stats_avg["DER"] = stats_avg["diarization_error"] / stats_avg["speaker_scored"]
        results = {
            "test/DER": stats_avg['DER'],
            "test/speech_scored": stats_avg['speech_scored'],
            "test/speech_miss": stats_avg['speech_miss'],
            "test/speech_falarm": stats_avg['speech_falarm'],
            "test/speaker_scored": stats_avg['speaker_scored'],
            "test/speaker_miss": stats_avg['speaker_miss'],
            "test/speaker_falarm": stats_avg['speaker_falarm'],
            "test/speaker_error": stats_avg['speaker_error'],
            "test/diarization_error": stats_avg['diarization_error'],
            "test/frames": stats_avg['frames'],
            "test/speaker_miss_rate": stats_avg['speaker_miss'] / stats_avg['speaker_scored'],
            "test/speaker_falarm_rate": stats_avg['speaker_falarm'] / stats_avg['speaker_scored'],
            "test/speaker_error_rate": stats_avg['speaker_error'] / stats_avg['speaker_scored'],
        }
        pprint(results)
        self.logger.log_metrics(results)
        self.logger.log_hyperparams(self.hparams, results)
        
        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=False)


    def configure_optimizers(self):
        if self.scheduler is not None:
            return {"optimizer": self.opt, "lr_scheduler": {"scheduler": self.scheduler, "interval": "step"}}
        else:
            return {"optimizer": self.opt}

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.hparams["training"]["batch_size"],
            shuffle=self.hparams["training"]["shuffle"],
            num_workers=self.hparams["training"]["n_workers"],
            collate_fn=self.collate_func
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.hparams["training"]["batch_size"],
            shuffle=False,
            num_workers=self.hparams["training"]["n_workers"],
            collate_fn=self.collate_func
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.hparams["training"]["batch_size"],
            shuffle=False,
            num_workers=self.hparams["training"]["n_workers"],
            collate_fn=self.collate_func
        )
    