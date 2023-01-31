import torch
import cv2
import os
import numpy as np
import torchvision
import os
from PIL import Image
from typing import List, Any
import pandas as pd


class RsnaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int,
        is_test: bool,
        augments: Any,
        img_folder: str,
        category_aux_targets: List[str],
    ):
        """Common Dataset for RSNA competition
        Training and Scoring.
        """
        df["img_name"] = (
            (df["patient_id"].astype(str) + "_" + df["image_id"].astype(str) + ".png")
            if not is_test
            else df["patient_id"].astype(str)
            + "_"
            + df["image_id"].astype(str)
            + ".png"
        )
        df = df.reset_index(drop=True)
        self.df = df
        self.is_test = is_test
        self.augments = augments
        self.img_folder = img_folder
        self.img_size = img_size
        self.category_aux_targets = category_aux_targets
        self.numpy_to_tensor_transfom = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_folder, self.df["img_name"][idx])
        img = cv2.imread(img_path)

        if self.augments:
            img = self.augments(image=img)["image"]
        img = self.numpy_to_tensor_transfom(img.astype(np.uint8))

        if not self.is_test:
            target = self.df["cancer"][idx]
            target = torch.tensor(target, dtype=torch.float)
            cat_aux_targets = torch.as_tensor(
                self.df.iloc[idx][self.category_aux_targets]
            )
            return img, target, cat_aux_targets
        patient_id = self.df["patient_id"][idx]
        laterality = self.df["laterality"][idx]
        return img, patient_id, laterality
