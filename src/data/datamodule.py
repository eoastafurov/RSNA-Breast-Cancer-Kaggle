import os
from typing import Optional, Tuple

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
        self.train_df.age.fillna(self.train_df.age.mean(), inplace=True)
        self.train_df.age = pd.qcut(
            self.train_df.age, 10, labels=range(10), retbins=False
        ).astype(int)
        self.train_df[self.conf["category_aux_targets"]] = self.train_df[
            self.conf["category_aux_targets"]
        ].apply(LabelEncoder().fit_transform)

        self.inference_df = pd.read_csv(
            os.path.join(self.conf["dataset_root"], "test.csv")
        )

    def load_augments(self):
        Augments = getattr(augments_factory, self.conf["augments_name"])
        augments = Augments(**self.conf["augments_kwargs"])
        self.train_augments = augments.train_augments
        self.val_augments = augments.valid_augments

    def make_data_split(
        self, kfold_num_splits: int, random_seed: int, selected_split: int
    ) -> None:
        """Generates ['split'] column in self.train_df according to arguments.
        Uses twice: firstly to separate test_df needed for ensembling and
        secondly to separate train_df from val_df.

        Uses sklearn StratifiedKFold splitter under the hood.
        """
        patient_id_any_cancer = (
            self.train_df.groupby("patient_id").cancer.max().reset_index()
        )
        splits = list(
            StratifiedKFold(
                kfold_num_splits,
                shuffle=True,
                random_state=random_seed,
            ).split(patient_id_any_cancer.patient_id, patient_id_any_cancer.cancer)
        )
        train_indices, val_indices = splits[selected_split]
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

        # Firstly, get test df for ensemble
        self.make_data_split(
            kfold_num_splits=20,  # --> to get approx 5% of data
            random_seed=2021,  # --> we are fixing it to reproduce then
            selected_split=0,  # --> we are fixing it to reproduce then
        )
        self.test_df = self.train_df[self.train_df["split"] == "test"]
        self.train_df = self.train_df[
            self.train_df["split"] == "train"
        ]  # we must update self.train_df to separate train from test
        self.train_df = self.train_df.drop(["split"], axis=1)

        # Then, get actual train and val dataframes
        self.make_data_split(
            kfold_num_splits=self.conf["kfold_num_splits"],
            random_seed=self.conf["random_seed"],
            selected_split=self.conf["kfold_selected_split"],
        )
        self.train_df, self.val_df = (
            self.train_df[self.train_df["split"] == "train"],
            self.train_df[self.train_df["split"] == "test"],
        )
        report = (
            "Lengths:\n\ttrain_df: {}, val_df: {}, test_df: {}\nTarget rates:\n\t"
            + "train_df: {}, val_df: {}, test_df: {}"
        ).format(
            len(self.train_df),
            len(self.val_df),
            len(self.test_df),
            round(self.train_df["cancer"].mean(), 4),
            round(self.val_df["cancer"].mean(), 4),
            round(self.test_df["cancer"].mean(), 4),
        )
        print(report)

        # Define datasets in self
        self.train_dataset = RsnaDataset(
            df=self.train_df,
            img_size=self.conf["img_size"],
            img_folder=self.conf["img_folder"],
            augments=self.train_augments,
            is_test=False,
            category_aux_targets=self.conf["category_aux_targets"],
        )
        self.val_dataset = RsnaDataset(
            df=self.val_df,
            img_size=self.conf["img_size"],
            img_folder=self.conf["img_folder"],
            augments=self.val_augments,
            is_test=False,
            category_aux_targets=self.conf["category_aux_targets"],
        )
        self.test_dataset = RsnaDataset(
            df=self.test_df,
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
        # self.inference_dataset = RsnaDataset(
        #     df=self.test_df,
        #     img_size=self.conf["img_size"],
        #     img_folder=self.conf["img_folder"],
        #     augments=self.val_augments,
        #     is_test=True,
        #     category_aux_targets=self.conf["category_aux_targets"],
        # )

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
