import os
import torch
import torch.nn.functional as F
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
        self.loss_func1 = pit_loss_multispk
        self.loss_func2 = standard_loss
        self.max_spks = self.hparams["data"]["max_speakers"]
        self.label_delay = self.hparams["data"]["label_delay"]

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
        clip_lengths = [x.shape[0] for x in feats]
        n_spks = [l.shape[1] for l in labels]
        max_spk = max(n_spks)
        labels = [F.pad(l, (0,max_spk-l.shape[1]), "constant", 0) for l in labels]

        # perm labels
        labels = torch.nn.utils.rnn.pad_sequence(labels, padding_value=0, batch_first=True)
        B, T, max_spk = labels.shape
        frame_index = torch.arange(1, T + 1).to(feats[0]).unsqueeze(0).unsqueeze(-1)
        labels_index = (frame_index * labels)
        labels_index = labels_index.masked_fill_(labels_index == 0, torch.inf)

        # [B, max_spk]
        sort_index = torch.argsort(torch.min(labels_index, dim=1)[0], dim=1)
        labels = labels[torch.arange(B).unsqueeze(1), :, sort_index].transpose(-1, -2)

        labels_silence = torch.ones([labels.shape[0], labels.shape[1]]).to(labels) - labels.max(-1)[0]
        labels = torch.cat([labels_silence.unsqueeze(-1), labels], dim=-1)

        labels_nonespk = torch.zeros([B, T, 1]).to(labels)
        labels = torch.cat([labels, labels_nonespk], dim=-1)

        labels = [l[:ilen, :nspk+2] for l, ilen, nspk in zip(labels, clip_lengths, n_spks)]

        # Transform feats to MelSpectrum
        preds, emb_loss, embs, attractors = self.detect(feats, labels, clip_lengths)

        labels_spks = [l[:, 1:-1] for l in labels]
        preds_spks = [p[:, 1:-1] for p in preds]

        max_nspk = max(n_spks)
        labels_spks_pad = pad_labels(labels_spks, max_nspk)
        preds_spks_pad = pad_preds(preds_spks, max_nspk)
        perm_labels_spks = self.loss_func1(preds_spks_pad, labels_spks_pad, n_spks)
        perm_labels = [torch.cat([l[:, 0].unsqueeze(-1), l_perm], dim=-1) for l, l_perm in zip(labels, perm_labels_spks)]
        perm_labels = [torch.cat([l_perm, l[:, -1].unsqueeze(-1)], dim=-1) for l ,l_perm in zip(labels, perm_labels)]
        pit_loss = self.loss_func2(preds, perm_labels, label_delay=self.label_delay)

        # Calculate the total loss
        tot_loss = pit_loss + emb_loss
        self.log("train/lr", self.opt.param_groups[-1]["lr"], prog_bar=True, sync_dist=True)
        self.log("train/pit_loss", pit_loss, sync_dist=True)
        self.log("train/emb_loss", emb_loss, sync_dist=True)
        self.log("train/tot_loss", tot_loss, sync_dist=True)

        return tot_loss

    def validation_step(self, batch, batch_index):
        feats, labels, _ = batch
        # Model pred
        clip_lengths = [x.shape[0] for x in feats]
        n_spks = [l.shape[1] for l in labels]
        max_spk = max(n_spks)
        labels = [F.pad(l, (0,max_spk-l.shape[1]), "constant", 0) for l in labels]

        # perm labels
        labels = torch.nn.utils.rnn.pad_sequence(labels, padding_value=0, batch_first=True)
        B, T, max_spk = labels.shape
        frame_index = torch.arange(1, T + 1).to(feats[0]).unsqueeze(0).unsqueeze(-1)
        labels_index = (frame_index * labels)
        labels_index = labels_index.masked_fill_(labels_index == 0, torch.inf)

        # [B, max_spk]
        sort_index = torch.argsort(torch.min(labels_index, dim=1)[0], dim=1)
        labels = labels[torch.arange(B).unsqueeze(1), :, sort_index].transpose(-1, -2)

        labels_silence = torch.ones([labels.shape[0], labels.shape[1]]).to(labels) - labels.max(-1)[0]
        labels = torch.cat([labels_silence.unsqueeze(-1), labels], dim=-1)
        
        labels_nonespk = torch.zeros([B, T, 1]).to(labels)
        labels = torch.cat([labels, labels_nonespk], dim=-1)

        labels = [l[:ilen, :nspk+2] for l, ilen, nspk in zip(labels, clip_lengths, n_spks)]

        preds, emb_loss, embs, attractors = self.detect(feats, labels, clip_lengths)

        labels_spks = [l[:, 1:-1] for l in labels]
        preds_spks = [p[:, 1:-1] for p in preds]

        max_nspk = max(n_spks)
        labels_spks_pad = pad_labels(labels_spks, max_nspk)
        preds_spks_pad = pad_preds(preds_spks, max_nspk)
        perm_labels_spks = self.loss_func1(preds_spks_pad, labels_spks_pad, n_spks)
        perm_labels = [torch.cat([l[:, 0].unsqueeze(-1), l_perm], dim=-1) for l, l_perm in zip(labels, perm_labels_spks)]
        perm_labels = [torch.cat([l_perm, l[:, -1].unsqueeze(-1)], dim=-1) for l ,l_perm in zip(labels, perm_labels)]
        pit_loss = self.loss_func2(preds, perm_labels, label_delay=self.label_delay)

        # Calculate the total loss
        tot_loss = pit_loss + emb_loss

        # Metrics
        preds_realspk = [p[:, 1:-1] for p in preds]
        labels_realspk = [l[:, 1:-1] for l in perm_labels]
        stats = report_diarization_error(preds_realspk, labels_realspk, label_delay=self.label_delay)

        self.log("val/pit_loss", pit_loss, sync_dist=True)
        self.log("val/emb_loss", emb_loss, sync_dist=True)
        self.log("val/tot_loss", tot_loss, sync_dist=True)

        stats["val_loss"] = [tot_loss.item()]
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

        # self.scheduler.step(stats_avg["val_loss"])

    def test_step(self, batch, batch_index):
        feats, labels, rec = batch
        # Model pred
        clip_lengths = [x.shape[0] for x in feats]
        n_spks = [l.shape[1] for l in labels]
        max_spk = max(n_spks)
        labels = [F.pad(l, (0,max_spk-l.shape[1]), "constant", 0) for l in labels]

        # perm labels
        labels = torch.nn.utils.rnn.pad_sequence(labels, padding_value=0, batch_first=True)
        B, T, max_spk = labels.shape
        frame_index = torch.arange(1, T + 1).to(feats[0]).unsqueeze(0).unsqueeze(-1)
        labels_index = (frame_index * labels)
        labels_index = labels_index.masked_fill_(labels_index == 0, torch.inf)

        # [B, max_spk]
        sort_index = torch.argsort(torch.min(labels_index, dim=1)[0], dim=1)
        labels = labels[torch.arange(B).unsqueeze(1), :, sort_index].transpose(-1, -2)

        labels_silence = torch.ones([labels.shape[0], labels.shape[1]]).to(labels) - labels.max(-1)[0]
        labels = torch.cat([labels_silence.unsqueeze(-1), labels], dim=-1)

        labels_nonespk = torch.zeros([B, T, 1]).to(labels)
        labels = torch.cat([labels, labels_nonespk], dim=-1)

        labels = [l[:ilen, :nspk+2] for l, ilen, nspk in zip(labels, clip_lengths, n_spks)]
        
        preds, embs, attractors = self.model.test(feats, clip_lengths, self.max_spks + 2)
        # feats_40s = [feats[0][:400]]
        # preds_40s, embs, attractors = self.model.test(feats_40s, [400], self.max_spks + 2)

        # print(preds[0][:390])
        # print(preds_40s[0][:390])
        # print((abs((preds[0][:390] - preds_40s[0][:390]))<1e-6).all())
        
        labels_spks = [l[:, 1:-1] for l in labels]
        preds_spks = [p[:, 1:nspk + 1] for p, nspk in zip(preds, n_spks)]

        max_nspk = max(n_spks)
        labels_spks_pad = pad_labels(labels_spks, max_nspk)
        preds_spks_pad = pad_preds(preds_spks, max_nspk)
        perm_labels_spks = self.loss_func1(preds_spks_pad, labels_spks_pad, n_spks)
        perm_labels = [torch.cat([l[:, 0].unsqueeze(-1), l_perm], dim=-1) for l, l_perm in zip(labels, perm_labels_spks)]
        perm_labels = [torch.cat([l_perm, l[:, -1].unsqueeze(-1)], dim=-1) for l ,l_perm in zip(labels, perm_labels)]
        # pit_loss = self.loss_func2(preds, perm_labels, label_delay=self.label_delay)

        # Calculate the total loss
        # tot_loss = pit_loss + emb_loss

        # Metrics
        preds_realspk = [p[:, 1:self.max_spks + 1] for p, nspk in zip(preds, n_spks)]
        if preds_realspk[0].shape[1] < self.max_spks:
            preds_realspk = pad_preds(preds_realspk, self.max_spks)
        labels_realspk = [l[:, 1:-1] for l in perm_labels]
        if labels_realspk[0].shape[1] < self.max_spks:
            labels_realspk = pad_preds(labels_realspk, self.max_spks)
        

        # preds_realspk = [p[:, 1:nspk + 1] for p, nspk in zip(preds, n_spks)]
        # if preds_realspk[0].shape[1] < max_nspk:
        #     preds_realspk = pad_preds(preds_realspk, max_nspk)
        # labels_realspk = [l[:, 1:-1] for l in perm_labels]
        # print(rec)
        # print(clip_lengths[0])
        stats = report_diarization_error(preds_realspk, labels_realspk, label_delay=self.label_delay)

        label_delay = self.hparams["data"]["label_delay"]
        preds = [torch.cat([p[label_delay:, 1:-1], p[-1, 1:-1].unsqueeze(0).repeat(label_delay, 1)], dim=0)  for p in preds]
        num_spks = "all"
        version = "_10w_ver_204_ami_ver_114_dev"
        save_dir_parnt = f"./outputs/data/onl_{num_spks}spk_version{version}"
        for content in ["preds", "labels", "embs", "attractors"]:
            save_dir = os.path.join(save_dir_parnt, content)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        # np.save(f"./outputs/data/onl_{num_spks}spk_version{version}/preds/{rec[0]}.npy", preds[0].detach().cpu().numpy())
        # np.save(f"./outputs/data/onl_{num_spks}spk_version{version}/labels/label_{rec[0]}.npy", labels_realspk[0].detach().cpu().numpy())
        # np.save(f"./outputs/data/onl_{num_spks}spk_version{version}/embs/emb_{rec[0]}.npy", embs[0].detach().cpu().numpy())
        # np.save(f"./outputs/data/onl_{num_spks}spk_version{version}/attractors/attractor_{rec[0]}.npy", attractors[0].detach().cpu().numpy())
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
            "test/preliminary_DER": stats_avg['DER'],
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
            batch_size=1,
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
    