import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import sklearn
import torch
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

from src.data import augments as augments_factory
from src.data.dataset import RsnaDataset


class RsnaDataModule(pl.LightningDataModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf

    def prepare_data(self):
        self.train_df = pd.read_csv(
            os.path.join(self.conf["dataset_root"], "train.csv")
        )
        self.inference_df = pd.read_csv(
            os.path.join(self.conf["dataset_root"], "test.csv")
        )
        # for _df in [self.train_df, self.inference_df]:
        #     _df.age.fillna(_df.age.mean(), inplace=True)
        #     _df.age = pd.qcut(_df.age, 10, labels=range(10), retbins=False).astype(int)
        #     _df[self.conf["category_aux_targets"]] = _df[
        #         self.conf["category_aux_targets"]
        #     ].apply(LabelEncoder().fit_transform)
        self.train_df.age.fillna(self.train_df.age.mean(), inplace=True)
        self.train_df.age = pd.qcut(
            self.train_df.age, 10, labels=range(10), retbins=False
        ).astype(int)
        self.train_df[self.conf["category_aux_targets"]] = self.train_df[
            self.conf["category_aux_targets"]
        ].apply(LabelEncoder().fit_transform)

    def load_augments(self):
        Augments = getattr(augments_factory, self.conf["augments_name"])
        augments = Augments(**self.conf["augments_kwargs"])
        self.train_augments = augments.train_augments
        self.val_augments = augments.valid_augments

    def setup(self, stage: Optional[str] = None):
        self.load_augments()
        train_df, val_df = sklearn.model_selection.train_test_split(
            self.train_df,
            test_size=self.conf["val_ratio"],
            stratify=self.train_df["cancer"],
        )
        self.train_dataset = RsnaDataset(
            df=train_df,
            img_size=self.conf["img_size"],
            img_folder=self.conf["img_folder"],
            augments=self.train_augments,
            is_test=False,
            category_aux_targets=self.conf["category_aux_targets"],
        )
        self.val_dataset = RsnaDataset(
            df=val_df,
            img_size=self.conf["img_size"],
            img_folder=self.conf["img_folder"],
            augments=self.val_augments,
            is_test=False,
            category_aux_targets=self.conf["category_aux_targets"],
        )
        self.inference_dataset = RsnaDataset(
            df=self.inference_df,
            img_size=self.conf["img_size"],
            img_folder=self.conf["img_folder_inference"],
            augments=self.val_augments,
            is_test=True,
            category_aux_targets=self.conf["category_aux_targets"],
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.conf["batch_size"],
            shuffle=True,
            num_workers=self.conf["num_workers"],
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.conf["batch_size"],
            shuffle=False,
            num_workers=self.conf["num_workers"],
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.inference_dataset,
            batch_size=self.conf["batch_size"],
            shuffle=False,
            num_workers=self.conf["num_workers"],
        )
