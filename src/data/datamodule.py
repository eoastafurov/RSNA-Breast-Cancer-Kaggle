import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import sklearn
import torch
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from pytorch_lightning.utilities.seed import seed_everything

from src.data import augments as augments_factory
from src.data.dataset import RsnaDataset


class RsnaDataModule(pl.LightningDataModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        seed_everything(conf["random_seed"], True)

    def prepare_data(self):
        self.train_df = pd.read_csv(
            os.path.join(self.conf["dataset_root"], "train.csv")
        )
        self.inference_df = pd.read_csv(
            os.path.join(self.conf["dataset_root"], "test.csv")
        )
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

    def make_data_split(self):
        patient_id_any_cancer = (
            self.train_df.groupby("patient_id").cancer.max().reset_index()
        )
        splits = list(
            StratifiedKFold(
                self.conf["kfold_num_splits"],
                shuffle=True,
                random_state=self.conf["random_seed"],
            ).split(patient_id_any_cancer.patient_id, patient_id_any_cancer.cancer)
        )
        train_indices, val_indices = splits[self.conf["kfold_selected_split"]]
        train_indices = list(patient_id_any_cancer["patient_id"][train_indices])
        val_indices = list(patient_id_any_cancer["patient_id"][val_indices])

        s = set(train_indices)
        self.train_df["split"] = [
            "train" if p_id in s else "test"
            for p_id in list(self.train_df["patient_id"])
        ]
        print(
            self.train_df[self.train_df["split"] == "train"]["cancer"].mean(),
            self.train_df[self.train_df["split"] == "test"]["cancer"].mean(),
        )

    def setup(self, stage: Optional[str] = None):
        self.load_augments()
        self.make_data_split()
        # train_df, val_df = sklearn.model_selection.train_test_split(
        #     self.train_df,
        #     test_size=self.conf["val_ratio"],
        #     stratify=self.train_df["cancer"],
        #     random_state=self.conf["random_seed"],
        # )
        train_df, val_df = (
            self.train_df[self.train_df["split"] == "train"],
            self.train_df[self.train_df["split"] == "test"],
        )
        # _, very_small_val = sklearn.model_selection.train_test_split(
        #     val_df,
        #     test_size=0.25,
        #     stratify=val_df["cancer"],
        #     random_state=self.conf["random_seed"],
        # )

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
            # df=very_small_val,
            img_size=self.conf["img_size"],
            img_folder=self.conf["img_folder"],
            augments=self.val_augments,
            is_test=False,
            category_aux_targets=self.conf["category_aux_targets"],
        )
        # self.test_dataset = RsnaDataset(
        #     df=very_small_val,
        #     img_size=self.conf["img_size"],
        #     img_folder=self.conf["img_folder"],
        #     augments=self.val_augments,
        #     is_test=False,
        #     category_aux_targets=self.conf["category_aux_targets"],
        # )
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

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
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
