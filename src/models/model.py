import torch
import timm
import pytorch_lightning as pl
import torchmetrics
import numpy as np
from fastai.losses import CrossEntropyLossFlat
from src.metrics.competition_metrics import c_metrics, pfbeta_torch_thresh, pfbeta_torch
from copy import deepcopy
from transformers import get_cosine_schedule_with_warmup
from pytorch_lightning.utilities.seed import seed_everything


class RsnaTimmModel(pl.LightningModule):
    def __init__(self, conf):
        super(RsnaTimmModel, self).__init__()
        self.save_hyperparameters()
        self.conf = conf
        seed_everything(conf["random_seed"], True)
        # Model Architecture
        self.model = timm.create_model(
            conf["model_name"],
            pretrained=conf["pretrained"],
            num_classes=0,
            drop_rate=conf["dropout_config"]["drop_rate"],
            drop_path_rate=conf["dropout_config"]["drop_path_rate"],
        )
        self.nn_aux = torch.nn.ModuleList(
            [torch.nn.LazyLinear(n) for n in conf["aux_classes"]]
        )
        self.fc = torch.nn.LazyLinear(1)
        self.dropout = torch.nn.Dropout(p=conf["dropout_config"]["fc_dropout"])
        self.pooling = torch.nn.AvgPool1d(kernel_size=1)
        # Loss functions
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([conf["positive_weight"]]).float()
        )
        # self.loss_fn = SmoothBCEwLogits(
        #     pos_weight=torch.tensor([50]).float(), smoothing=0.1
        # )

        # Metric
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        if self.conf["enable_grad"]:
            output = self.model(x)
        else:
            with torch.no_grad():
                output = self.model(x)
        output = self.dropout(output)
        output = self.fc(output)

        aux = []
        for nn in self.nn_aux:
            tmp = nn(output)
            aux.append(tmp)
        return output, aux

    def training_step(self, batch, batch_idx):
        imgs, targets, cat_aux_targets = batch
        out, aux_out = self(imgs)
        out = out.view(-1)
        loss = self.loss_fn(out, targets)
        clean_loss = loss.clone().detach().cpu()
        self.log(
            "train_loss_clean",
            clean_loss,
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
            "clean_loss_train": clean_loss,
        }

    def validation_step(self, batch, batch_idx):
        imgs, targets, cat_aux_targets = batch
        out, aux_out = self(imgs)
        # out, aux_out = self(imgs.contigous())
        out = out.view(-1)
        loss = self.loss_fn(out, targets)
        clean_loss_val = loss.clone().detach().cpu()
        self.log(
            "val_loss_clean",
            clean_loss_val,
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
            "clean_loss_val": clean_loss_val,
        }

    def test_step(self, batch, batch_idx):
        imgs, targets, cat_aux_targets = batch
        out, aux_out = self(imgs)
        # out, aux_out = self(imgs.contigous())
        out = out.view(-1)
        loss = self.loss_fn(out, targets)
        clean_loss_test = loss.clone().detach().cpu()
        self.log(
            "test_loss_clean",
            clean_loss_test,
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
        cf1_thresh, _ = pfbeta_torch_thresh(
            preds=self.sigmoid(out).clone().detach().cpu(),
            labels=targets.clone().detach().cpu(),
        )

        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "test_cf1",
            cf1,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "test_cf1_thresh",
            cf1_thresh,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return {
            "test_loss": loss,
            "test_cf1": cf1,
            "test_cf1_thresh": cf1_thresh,
            "pred_probs": self.sigmoid(out).clone().detach().cpu(),
            "y_test": targets.clone().detach().cpu(),
            "clean_loss_test": clean_loss_test,
        }

    def validation_epoch_end(self, outputs):
        val_avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_avg_loss_clean = torch.stack([x["clean_loss_val"] for x in outputs]).mean()
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
            = VALID LOSS CLEAN: {5}
            = VALID cF1: {2}
            = VALID cF1_thresh {3}, thr value = {4}
        #{0}#""".format(
            "".join(["="] * 75),
            round(float(val_avg_loss), 3),
            round(float(cf1), 3),
            round(float(cf1_thresh), 3),
            round(float(threshold), 3),
            round(float(val_avg_loss_clean), 3),
        )
        print(report)

    def training_epoch_end(self, outputs):
        train_avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_avg_loss_clean = torch.stack(
            [x["clean_loss_train"] for x in outputs]
        ).mean()
        train_y_test = torch.cat([x["y_test"] for x in outputs])
        train_pred_probs = torch.cat([x["pred_probs"] for x in outputs])

        cf1 = pfbeta_torch(
            preds=train_pred_probs,
            labels=train_y_test,
        )
        cf1_thresh, threshold = pfbeta_torch_thresh(
            preds=train_pred_probs,
            labels=train_y_test,
        )
        report = """        #{0}#
            = TRAIN LOSS: {1}
            = TRAIN LOSS CLEAN: {5}
            = TRAIN cF1: {2}
            = TRAIN cF1_thresh {3}, thr value = {4}
        #{0}#""".format(
            "".join(["="] * 75),
            round(float(train_avg_loss) * self.conf["grad_accum_steps"], 3),
            round(float(cf1), 3),
            round(float(cf1_thresh), 3),
            round(float(threshold), 3),
            round(float(train_avg_loss_clean), 3),
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
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.conf["weight_decay"],
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        num_train_optimization_steps = self.conf["__total_training_steps__"]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.conf["lr"],
            betas=(0.9, 0.999),
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.conf["warmup_ratio"] * num_train_optimization_steps,
            num_training_steps=num_train_optimization_steps,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
