from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    IAAPerspective,
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    IAAPiecewiseAffine,
    RandomResizedCrop,
    IAASharpen,
    IAAEmboss,
    RandomBrightnessContrast,
    Flip,
    OneOf,
    Compose,
    Normalize,
    Cutout,
    CoarseDropout,
    ShiftScaleRotate,
    CenterCrop,
    Resize,
)
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2


class SimpleAugments:
    """Most simple augmentations"""

    def __init__(self, img_size):
        self.train_augments = Compose(
            [
                Resize(img_size, img_size, p=1.0),
                # A.Equalize(mode="cv", by_channels=True, p=0.05),
                # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.1),
                # A.InvertImg(p=0.05),
                # A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.05),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                # A.Superpixels(
                #     p_replace=0.1,
                #     n_segments=100,
                #     max_size=128,
                #     interpolation=1,
                #     always_apply=False,
                #     p=0.05,
                # ),
                # A.CoarseDropout(
                #     max_holes=25,
                #     max_height=img_size // 16,
                #     max_width=img_size // 16,
                #     p=0.2,
                # ),
                A.GridDistortion(
                    num_steps=2,
                    distort_limit=0.3,
                    p=0.2,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                # A.augmentations.transforms.Sharpen(
                #     alpha=(0.05, 0.1), lightness=(0.25, 0.5), always_apply=False, p=0.1
                # ),
                # A.RandomBrightnessContrast(
                #     brightness_limit=(-0.05, 0.05),
                #     contrast_limit=(-0.2, 0.2),
                #     p=0.5,
                # ),
                # A.RandomCrop(
                #     height=int(img_size * 0.95),
                #     width=int(img_size * 0.95),
                #     always_apply=False,
                #     p=0.5,
                # ),
                Resize(img_size, img_size, p=1.0),
                ######### New version ##########
                # A.Normalize(mean=0, std=1),
                # ToTensorV2(),
            ],
        )

        self.valid_augments = Compose(
            [
                Resize(img_size, img_size, p=1.0),
                ######### New version ##########
                # A.Normalize(mean=0, std=1),
                # ToTensorV2(),
            ],
        )

        # self.multiple_times_test_augments = Compose(
        #     [
        #         Resize(img_size, img_size, p=1.0),
        #         A.augmentations.transforms.Sharpen(
        #             alpha=(0.2, 0.21), lightness=(0.25, 0.5), always_apply=False, p=0.5
        #         ),
        #         A.Equalize(mode="cv", by_channels=True, p=0.5),
        #         A.HorizontalFlip(p=0.5),
        #         A.VerticalFlip(p=0.1),
        #         Resize(img_size, img_size, p=1.0),
        #     ],
        # )


# from albumentations import (
#     HorizontalFlip,
#     VerticalFlip,
#     IAAPerspective,
#     ShiftScaleRotate,
#     CLAHE,
#     RandomRotate90,
#     Transpose,
#     ShiftScaleRotate,
#     Blur,
#     OpticalDistortion,
#     GridDistortion,
#     HueSaturationValue,
#     IAAAdditiveGaussianNoise,
#     GaussNoise,
#     MotionBlur,
#     MedianBlur,
#     IAAPiecewiseAffine,
#     RandomResizedCrop,
#     IAASharpen,
#     IAAEmboss,
#     RandomBrightnessContrast,
#     Flip,
#     OneOf,
#     Compose,
#     Normalize,
#     Cutout,
#     CoarseDropout,
#     ShiftScaleRotate,
#     CenterCrop,
#     Resize,
# )
# from albumentations.pytorch import ToTensorV2
# import albumentations as A
# import cv2


# class SimpleAugments:
#     """Most simple augmentations"""

#     def __init__(self, img_size):
#         self.train_augments = Compose(
#             [
#                 Resize(img_size, img_size, p=1.0),
#                 A.Equalize(mode="cv", by_channels=True, p=0.1),
#                 A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.1),
#                 A.InvertImg(p=0.05),
#                 A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.05),
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.1),
#                 A.Superpixels(
#                     p_replace=0.1,
#                     n_segments=100,
#                     max_size=128,
#                     interpolation=1,
#                     always_apply=False,
#                     p=0.05,
#                 ),
#                 A.CoarseDropout(
#                     max_holes=48,
#                     max_height=img_size // 16,
#                     max_width=img_size // 16,
#                     p=0.1,
#                 ),
#                 A.GridDistortion(
#                     num_steps=5,
#                     distort_limit=0.5,
#                     p=0.25,
#                     border_mode=cv2.BORDER_CONSTANT,
#                 ),
#                 A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
#                 A.augmentations.transforms.Sharpen(
#                     alpha=(0.05, 0.1), lightness=(0.25, 0.5), always_apply=False, p=0.25
#                 ),
#                 A.RandomBrightnessContrast(
#                     brightness_limit=(-0.05, 0.05),
#                     contrast_limit=(-0.33, 0.33),
#                     p=0.5,
#                 ),
#                 A.RandomCrop(
#                     height=int(img_size * 0.7),
#                     width=int(img_size * 0.7),
#                     always_apply=False,
#                     p=0.2,
#                 ),
#                 # Normalize(
#                 #     mean=[0.485, 0.456, 0.406],
#                 #     std=[0.229, 0.224, 0.225],
#                 #     max_pixel_value=255.0,
#                 #     p=1.0,
#                 # ),
#                 Resize(img_size, img_size, p=1.0),
#             ],
#         )

#         self.valid_augments = Compose(
#             [
#                 Resize(img_size, img_size, p=1.0),
#                 # Normalize(
#                 #     mean=[0.485, 0.456, 0.406],
#                 #     std=[0.229, 0.224, 0.225],
#                 #     max_pixel_value=255.0,
#                 #     p=1.0,
#                 # ),
#             ],
#         )

#         self.multiple_times_test_augments = Compose(
#             [
#                 Resize(img_size, img_size, p=1.0),
#                 A.augmentations.transforms.Sharpen(
#                     alpha=(0.2, 0.21), lightness=(0.25, 0.5), always_apply=False, p=0.5
#                 ),
#                 A.Equalize(mode="cv", by_channels=True, p=0.5),
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.1),
#                 Resize(img_size, img_size, p=1.0),
#             ],
#         )
