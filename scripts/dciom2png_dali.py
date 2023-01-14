import argparse
import glob
import logging
import os
import shutil

import cv2
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import pydicom

import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type
from nvidia.dali.types import DALIDataType
from pydicom.filebase import DicomBytesIO
from tqdm.notebook import tqdm
import dicomsdl


def configure_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--max_images", type=int, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--destination", type=str, required=True)
    parser.add_argument("--j2k_dir", type=str, required=True)
    parser.add_argument("--njobs", type=int, required=False, default=2)
    parser.add_argument("--nchunks", type=int, required=False, default=4)
    return parser


def dicomsdl_to_numpy_image(dicom, index=0):
    info = dicom.getPixelDataInfo()
    dtype = info["dtype"]
    if info["SamplesPerPixel"] != 1:
        raise RuntimeError("SamplesPerPixel != 1")
    else:
        shape = [info["Rows"], info["Cols"]]
    outarr = np.empty(shape, dtype=dtype)
    dicom.copyFrameData(index, outarr)
    return outarr


def load_img_dicomsdl(f):
    return dicomsdl_to_numpy_image(dicomsdl.open(f))


def process_rest_func(f, size=512, save_folder=""):
    patient = f.split("/")[-2]
    image = f.split("/")[-1][:-4]

    dicom = pydicom.dcmread(f)

    if (
        dicom.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.4.90"
    ):  # ALREADY PROCESSED
        return

    try:
        img = load_img_dicomsdl(f)
    except:
        img = dicom.pixel_array

    img = (img - img.min()) / (img.max() - img.min())

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    img = cv2.resize(img, (size, size))

    cv2.imwrite(save_folder + f"{patient}_{image}.png", (img * 255).astype(np.uint8))


def convert_dicom_to_j2k(file, save_folder=""):
    patient = file.split("/")[-2]
    image = file.split("/")[-1][:-4]
    dcmfile = pydicom.dcmread(file)
    if dcmfile.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.4.90":
        with open(file, "rb") as fp:
            raw = DicomBytesIO(fp.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(
            b"\x00\x00\x00\x0C"
        )  # <---- the jpeg2000 header info we're looking for
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        with open(save_folder + f"{patient}_{image}.jp2", "wb") as binary_file:
            binary_file.write(hackedbitstream)


@pipeline_def
def j2k_decode_pipeline(j2kfiles):
    jpegs, _ = fn.readers.file(files=j2kfiles)
    images = fn.experimental.decoders.image(
        jpegs, device="mixed", output_type=types.ANY_DATA, dtype=DALIDataType.UINT16
    )
    return images


def main(args):
    images_paths = glob.glob(f"{args.source}*/*.dcm")[:args.max_images]
    print("Number of images :", len(images_paths))
    print("Making savedirs tree")
    os.makedirs(args.destination, exist_ok=True)
    chunks = [
        (
            len(images_paths) / args.nchunks * k,
            len(images_paths) / args.nchunks * (k + 1),
        )
        for k in range(args.nchunks)
    ]
    chunks = np.array(chunks).astype(int)
    j2k_dir = args.j2k_dir

    for i, chunk in tqdm(enumerate(chunks)):
        print(f"Starting {i+1} chunk")

        os.makedirs(j2k_dir, exist_ok=True)
        _ = Parallel(n_jobs=args.njobs)(
            delayed(convert_dicom_to_j2k)(fname, save_folder=j2k_dir)
            for fname in images_paths[chunk[0] : chunk[1]]
        )

        j2kfiles = glob.glob(j2k_dir + "*.jp2")
        if not len(j2kfiles):
            continue

        pipe = j2k_decode_pipeline(
            j2kfiles, batch_size=1, num_threads=2, device_id=0, debug=True
        )
        pipe.build()

        for i, f in tqdm(enumerate(j2kfiles)):
            patient, image = f.split("/")[-1][:-4].split("_")
            dicom = pydicom.dcmread(args.source + f"{patient}/{image}.dcm")

            # Dali -> Torch
            out = pipe.run()
            img = out[0][0]
            img_torch = torch.empty(img.shape(), dtype=torch.int16, device="cuda")
            feed_ndarray(
                img, img_torch, cuda_stream=torch.cuda.current_stream(device=0)
            )
            img = img_torch.float()

            # Scale, resize, invert on GPU !
            min_, max_ = img.min(), img.max()
            img = (img - min_) / (max_ - min_)
            if args.img_size:
                img = F.interpolate(
                    img.view(1, 1, img.size(0), img.size(1)),
                    (args.img_size, args.img_size),
                    mode="bilinear",
                )[0, 0]
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                img = 1 - img

            # Back to CPU + SAVE
            img = (img * 255).cpu().numpy().astype(np.uint8)
            cv2.imwrite(args.destination + f"{patient}_{image}.png", img)
        shutil.rmtree(j2k_dir, ignore_errors=True)

    print("Done with GPU!")
    print("Starting processing the rest on CPU..")

    _ = Parallel(n_jobs=int(args.njobs))(
        delayed(process_rest_func)(
            img, size=args.img_size, save_folder=args.destination
        )
        for img in tqdm(images_paths)
    )

    print("Done!")


if __name__ == "__main__":
    """Example:
    >>> python dciom2png_dali.py --img_size 1024 --source /data/rsna/test_images/ \
            --destination /home/toomuch/rsna/to_delete/output/ \
            --j2k_dir /home/toomuch/rsna/to_delete/j2k/ \
    >>> python dciom2png_dali.py --img_size 1024 --max_images 100 --source /home/jovyan/Datasets/rsna/train_images/ --destination /home/jovyan/Datasets/rsna/output/ --j2k_dir /home/jovyan/Datasets/rsna/j2k/ --njobs 4 --nchunks 1
    """
    args = configure_parser().parse_args()
    main(args)
