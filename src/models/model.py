import torch
import timm
import pytorch_lightning as pl
import torchmetrics
import numpy as np
from fastai.losses import CrossEntropyLossFlat
from src.metrics.competition_metrics import c_metrics, pfbeta_torch_thresh, pfbeta_torch
from torch.nn.modules.loss import _WeightedLoss
from copy import deepcopy


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0, pos_weight=None):
        super().__init__(weight=weight, reduction=reduction)
        self.save_hyperparameters()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    @staticmethod
    def _smooth(targets, n_labels, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, self.weight, pos_weight=self.pos_weight.to("cuda")
        )
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


class RsnaTimmModel(pl.LightningModule):
    def __init__(self, conf):
        super(RsnaTimmModel, self).__init__()
        self.conf = conf
        # Model Architecture
        self.model = timm.create_model(
            conf["model_name"],
            pretrained=conf['pretrained'],
            num_classes=0,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )
        self.nn_aux = torch.nn.ModuleList(
            [torch.nn.LazyLinear(n) for n in conf["aux_classes"]]
        )
        self.fc = torch.nn.LazyLinear(1)
        self.dropout = torch.nn.Dropout(p=conf["fc_dropout"])
        self.pooling = torch.nn.AvgPool1d(kernel_size=1)
        # Loss functions
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50]).float())
        # self.loss_fn = SmoothBCEwLogits(
        #     pos_weight=torch.tensor([50]).float(), smoothing=0.1
        # )

        # Metric
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        output = self.model(x)
        output = self.dropout(output)
        output = self.fc(output)

        aux = []
        for nn in self.nn_aux:
            # print(nn)
            tmp = nn(output)
            # print(tmp.shape)
            aux.append(tmp)
        return output, aux

    def training_step(self, batch, batch_idx):
        imgs, targets, cat_aux_targets = batch
        out, aux_out = self(imgs)
        out = out.view(-1)
        loss = self.loss_fn(out, targets)
        self.log(
            "train_loss_clear",
            loss.clone(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        aux_loss = torch.mean(
            torch.stack(
                [
                    torch.nn.functional.cross_entropy(aux_out[i], cat_aux_targets[:, i])
                    for i in range(cat_aux_targets.shape[-1])
                ]
            )
        )
        loss = loss + self.conf["aux_loss_weight"] * aux_loss

        cf1 = pfbeta_torch(
            preds=self.sigmoid(out).clone().detach().cpu(),
            labels=targets.clone().detach().cpu(),
        )
        cf1_thresh, threshold = pfbeta_torch_thresh(
            preds=self.sigmoid(out).clone().detach().cpu(),
            labels=targets.clone().detach().cpu(),
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "train_cf1",
            cf1,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train_cf1_thresh",
            cf1_thresh,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train_cf1_threshold_value",
            threshold,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        logs = {"train_loss": loss}
        return {
            "loss": loss,
            "log": logs,
            "pred_probs": self.sigmoid(out).clone().detach().cpu(),
            "y_test": targets.clone().detach().cpu(),
        }

    def validation_step(self, batch, batch_idx):
        imgs, targets, cat_aux_targets = batch
        out, aux_out = self(imgs)
        out = out.view(-1)
        loss = self.loss_fn(out, targets)
        aux_loss = torch.mean(
            torch.stack(
                [
                    torch.nn.functional.cross_entropy(aux_out[i], cat_aux_targets[:, i])
                    for i in range(cat_aux_targets.shape[-1])
                ]
            )
        )
        loss = loss + self.conf["aux_loss_weight"] * aux_loss

        cf1 = pfbeta_torch(
            preds=self.sigmoid(out).clone().detach().cpu(),
            labels=targets.clone().detach().cpu(),
        )
        cf1_thresh, _ = pfbeta_torch_thresh(
            preds=self.sigmoid(out).clone().detach().cpu(),
            labels=targets.clone().detach().cpu(),
        )

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_cf1",
            cf1,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_cf1_thresh",
            cf1_thresh,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return {
            "val_loss": loss,
            "val_cf1": cf1,
            "val_cf1_thresh": cf1_thresh,
            "pred_probs": self.sigmoid(out).clone().detach().cpu(),
            "y_test": targets.clone().detach().cpu(),
        }

    def validation_epoch_end(self, outputs):
        val_avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_y_test = torch.cat([x["y_test"] for x in outputs])
        val_pred_probs = torch.cat([x["pred_probs"] for x in outputs])

        cf1 = pfbeta_torch(
            preds=val_pred_probs,
            labels=val_y_test,
        )
        cf1_thresh, threshold = pfbeta_torch_thresh(
            preds=val_pred_probs,
            labels=val_y_test,
        )

        report = """        #{0}#
            = VALID LOSS: {1}
            = VALID cF1: {2}
            = VALID cF1_thresh {3}, thr value = {4}
        #{0}#""".format(
            "".join(["="] * 75),
            round(float(val_avg_loss), 3),
            round(float(cf1), 3),
            round(float(cf1_thresh), 3),
            round(float(threshold), 3),
        )
        print(report)

    def training_epoch_end(self, outputs):
        train_avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_y_test = torch.cat([x["y_test"] for x in outputs])
        train_pred_probs = torch.cat([x["pred_probs"] for x in outputs])

        cf1 = pfbeta_torch(
            preds=train_pred_probs,
            labels=train_y_test,
        )
        cf1_thresh, _ = pfbeta_torch_thresh(
            preds=train_pred_probs,
            labels=train_y_test,
        )
        report = """        #{0}#
            = TRAIN LOSS: {1}
            = TRAIN cF1: {2}
            = TRAIN cF1_thresh {3}
        #{0}#""".format(
            "".join(["="] * 75),
            round(float(train_avg_loss), 3),
            round(float(cf1), 3),
            round(float(cf1_thresh), 3),
            # round(float(cm["c_precision"]), 3),
            # round(float(cm["c_recall"]), 3),
        )
        print(report)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, patient_ids, lateralitys = batch
        out, _ = self(imgs)
        probs = self.sigmoid(out)
        return probs, patient_ids, lateralitys

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.conf["lr"],
            betas=(0.9, 0.999),
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.3,
            patience=2,
            threshold=1e-2,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_cf1",
        }
        return [optimizer], [lr_dict]
