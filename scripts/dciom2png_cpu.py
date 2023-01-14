import pydicom
from pathlib import Path
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
import multiprocessing as mp


def configure_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--size", type=int)
    return parser


def dicom_file_to_ary(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def process_directory(directory_path: str, dest: str, size: int):
    parent_directory = str(directory_path).split("/")[-1]
    os.mkdir(os.path.join(dest, parent_directory))
    for image_path in directory_path.iterdir():
        processed_ary = dicom_file_to_ary(image_path)
        im = Image.fromarray(processed_ary).resize((size, size))
        im.save(os.path.join(dest, parent_directory, f"{image_path.stem}.png"))


if __name__ == "__main__":
    """Example:
    >>> python dciom2png_cpu.py --source /data/rsna/train_images --dest /data/rsna/train_images_png_512 --size 512
    """
    args = configure_parser().parse_args()
    directories = list(Path(args.source).iterdir())

    with mp.Pool(32) as p:
        p.starmap(
            process_directory,
            tqdm([(d, args.dest, int(args.size)) for d in directories]),
        )
