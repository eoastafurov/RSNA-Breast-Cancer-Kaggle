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

RANDOM_STATE = 44
seed_everything(RANDOM_STATE, True)


RUN_NAME = "tf_efficientnetv2_m_4_5_fold_1024"
DESC = "?"
# MODEL_NAME = "tf_efficientnetv2_l"
# MODEL_NAME = "efficientnet_b0"
# MODEL_NAME = "tf_efficientnetv2_s"
MODEL_NAME = "tf_efficientnetv2_m"
# MODEL_NAME = "efficientnet_b4"
# MODEL_NAME = "seresnext50_32x4d"


conf = {
    "random_seed": RANDOM_STATE,
    "dataset_root": "/data/rsna",
    "augments_name": "SimpleAugments",
    "augments_kwargs": {"img_size": 1024},
    "kfold_num_splits": 5,
    "kfold_selected_split": 4,
    "positive_weight": 50,
    "img_size": 1024,
    "lr": 3e-4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.1,
    "dropout_config": {
        "drop_rate": 0.3,
        "drop_path_rate": 0.2,
        "fc_dropout": 0.0000000001,
    },
    "aux_loss_weight": 0.00000001,  # TODO: Try to disable it after main loop
    "img_folder": "/data/rsna/train_images_1024_dali",
    "img_folder_inference": "?",
    "batch_size": 24,
    "grad_accum_steps": 4,
    "epochs": 12,
    "num_workers": 16,
    "model_name": MODEL_NAME,
    "pretrained": True,
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
}


if __name__ == "__main__":
    dm = RsnaDataModule(conf)
    dm.prepare_data()
    dm.setup()
    conf["__total_training_steps__"] = int(
        len(dm.train_dataloader()) * conf["epochs"] / conf["grad_accum_steps"]
    )
    print(conf["__total_training_steps__"])
    # print(
    #     "Total train samples: {}\nTotal train batches: {}\nTotal valid samples: {}\n Total valid batches: {}".format(
    #         len(dm.train_dataloader()) * conf["batch_size"],
    #         len(dm.train_dataloader()),
    #         len(dm.val_dataloader()) * conf["batch_size"],
    #         len(dm.val_dataloader()),
    #     )
    # )
    model = RsnaTimmModel(conf)
    # print(
    #     model.load_state_dict(
    #         torch.load(
    #             "/home/jovyan/rsna/runs/tf_efficientnetv2_s_1024/tf_efficientnetv2_s_1024-epoch=06-val_loss=1.456-val_cf1=0.057.ckpt"
    #         )["state_dict"]
    #     )
    # )

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
    lr_monitor = LearningRateMonitor(logging_interval="step")

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
        max_epochs=conf["epochs"],
        callbacks=[
            checkpoint_callback,
            progress_bar,
            lr_monitor,
        ],
        check_val_every_n_epoch=1,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=10.0,
        val_check_interval=0.33,
        #
        accumulate_grad_batches={0: conf["grad_accum_steps"]},
        # profiler="advanced",
        # track_grad_norm=2,
        # detect_anomaly=True,
        # overfit_batches=5
    )

    trainer.fit(model, datamodule=dm)
