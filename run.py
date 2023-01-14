from typing import List, Tuple
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging,
    GradientAccumulationScheduler,
)
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.strategies import (
    DDPStrategy,
    DataParallelStrategy,
    DeepSpeedStrategy,
)

from src.models.model import RsnaTimmModel
from src.data.datamodule import RsnaDataModule


from pytorch_lightning.utilities.seed import seed_everything

seed_everything(44, True)


RUN_NAME = ""
DESC = ""
# MODEL_NAME = "tf_efficientnetv2_l"
MODEL_NAME = "tf_efficientnetv2_s"
# MODEL_NAME = "seresnext50_32x4d"


conf = {
    "dataset_root": "/home/jovyan/Datasets/rsna",
    "augments_name": "SimpleAugments",
    "augments_kwargs": {"img_size": 1024},
    "val_ratio": 0.2,
    "img_size": 1024,
    "img_folder": "/home/jovyan/Datasets/rsna/train_images_1024",
    "batch_size": 8,
    "num_workers": 16,
    "model_name": MODEL_NAME,
    "pretrained": True,
    "fc_dropout": 0.65,
    "lr": 5e-5,
    "category_aux_targets": [
        "site_id",
        "laterality",
        "view",
        "implant",
        "biopsy",
        "invasive",
        "BIRADS",
        "density",
        "difficult_negative_case",
        "machine_id",
        "age",
    ],
    "aux_classes": [2, 2, 6, 2, 2, 2, 4, 5, 2, 10, 10],
    "aux_loss_weight": 1.0,  # ?
}


if __name__ == "__main__":
    dm = RsnaDataModule(conf)
    dm.prepare_data()
    dm.setup()
    # print(dir(dm))
    model = RsnaTimmModel(conf)
    # print(model.load_state_dict(torch.load('/home/toomuch/rsna/runs/[READY 0.294] EffNet2_s_1024_AUX/tf_efficientnetv2_s_1024_debug-epoch=09-val_loss=2.238-val_cf1=0.062_cf1thr=0.3.ckpt')['state_dict']))

    logger = TensorBoardLogger("lightning_logs", name=RUN_NAME)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=50,
        # monitor="val_cf1_thresh",
        # mode="max",
        monitor="val_loss",
        mode="min",
        dirpath="runs/{}/".format(RUN_NAME),
        filename=RUN_NAME
        + "-{epoch:02d}-{val_loss:.3f}-{val_cf1:.3f}",  # -{val_cf1_thresh:.3f}",
        save_last=False,
        verbose=True,
        every_n_epochs=1,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=20,
        verbose=True,
        strict=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    accumulator = GradientAccumulationScheduler(scheduling={0: 16})

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        ),
        refresh_rate=1,
        leave=False,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # strategy=DDPStrategy(find_unused_parameters=False),
        precision=16,
        max_epochs=50,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            progress_bar,
            lr_monitor,
            # swa_callback,
        ],
        check_val_every_n_epoch=1,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=10.0,
        num_sanity_val_steps=10,
        #
        # profiler="advanced",
        # track_grad_norm=2,
        # detect_anomaly=True,
    )

    trainer.fit(model, datamodule=dm)
