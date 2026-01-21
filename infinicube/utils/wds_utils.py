# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import re
from pathlib import Path

import numpy as np
import webdataset as wds
from webdataset import WebDataset, non_empty

MY_IMAGE_EXTENSIONS = [
    "blp",
    "bmp",
    "dib",
    "bufr",
    "cur",
    "pcx",
    "dcx",
    "dds",
    "ps",
    "eps",
    "fit",
    "fits",
    "fli",
    "flc",
    "ftc",
    "ftu",
    "gbr",
    "gif",
    "grib",
    "h5",
    "hdf",
    "png",
    "apng",
    "jp2",
    "j2k",
    "jpc",
    "jpf",
    "jpx",
    "j2c",
    "icns",
    "ico",
    "im",
    "iim",
    "tif",
    "tiff",
    "jfif",
    "jpe",
    "jpg",
    "jpeg",
    "mpg",
    "mpeg",
    "msp",
    "pcd",
    "pxr",
    "pbm",
    "pgm",
    "ppm",
    "pnm",
    "psd",
    "bw",
    "rgb",
    "rgba",
    "sgi",
    "ras",
    "tga",
    "icb",
    "vda",
    "vst",
    "webp",
    "wmf",
    "emf",
    "xbm",
    "xpm",
]

my_imagespecs = {
    "npraw": ("numpy", None, None),
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),
    "l": ("numpy", "float", "l"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),
    "torchl8": ("torch", "uint8", "l"),
    "torchrgb8": ("torch", "uint8", "rgb"),
    "torchrgba8": ("torch", "uint8", "rgba"),
    "torchl": ("torch", "float", "l"),
    "torchrgb": ("torch", "float", "rgb"),
    "torch": ("torch", "float", "rgb"),
    "torchrgba": ("torch", "float", "rgba"),
    "pill": ("pil", None, "l"),
    "pil": ("pil", None, "rgb"),
    "pilrgb": ("pil", None, "rgb"),
    "pilrgba": ("pil", None, "rgba"),
}


class MyImageHandler:
    """Decode image data using the given `imagespec`.

    The `imagespec` specifies whether the image is decoded
    to numpy/torch/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - npraw: numpy None None
    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchl: torch float l
    - torchrgb: torch float rgb
    - torch: torch float rgb
    - torchrgba: torch float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba

    """

    def __init__(self, imagespec, extensions=MY_IMAGE_EXTENSIONS):
        """Create an image handler.

        :param imagespec: short string indicating the type of decoding
        :param extensions: list of extensions the image handler is invoked for
        """
        if imagespec not in list(my_imagespecs.keys()):
            raise ValueError(
                "Unknown imagespec: %s. \n\
                              If it is `npraw` (numpy raw), you shoud pip install git+https://github.com/yifanlu0227/webdataset.git rather than the official one"
                % imagespec
            )
        self.imagespec = imagespec.lower()
        self.extensions = extensions

    def __call__(self, key, data):
        """Perform image decoding.

        :param key: file name extension
        :param data: binary data
        """
        import PIL.Image

        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None
        imagespec = self.imagespec
        atype, etype, mode = my_imagespecs[imagespec]

        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            if mode is not None:
                img = img.convert(mode.upper())

        if atype == "pil":
            if mode == "l":
                img = img.convert("L")
                return img
            elif mode == "rgb":
                img = img.convert("RGB")
                return img
            elif mode == "rgba":
                img = img.convert("RGBA")
                return img
            else:
                raise ValueError("Unknown mode: %s" % mode)

        result = np.asarray(img)

        if etype == "float":
            result = result.astype(np.float32) / 255.0

        assert result.ndim in [2, 3], result.shape
        assert mode in ["l", "rgb", "rgba", None], mode

        if mode == "l":
            if result.ndim == 3:
                result = np.mean(result[:, :, :3], axis=2)
        elif mode == "rgb":
            if result.ndim == 2:
                result = np.repeat(result[:, :, np.newaxis], 3, axis=2)
            elif result.shape[2] == 4:
                result = result[:, :, :3]
        elif mode == "rgba":
            if result.ndim == 2:
                result = np.repeat(result[:, :, np.newaxis], 4, axis=2)
                result[:, :, 3] = 255
            elif result.shape[2] == 3:
                result = np.concatenate(
                    [result, 255 * np.ones(result.shape[:2])], axis=2
                )

        assert atype in ["numpy", "torch"], atype

        if atype == "numpy":
            return result
        elif atype == "torch":
            import torch

            if result.ndim == 3:
                return torch.from_numpy(result.transpose(2, 0, 1).copy())
            else:
                return torch.from_numpy(result.copy())

        return None


def my_imagehandler(imagespec, extensions=MY_IMAGE_EXTENSIONS):
    """Create an image handler.

    This is just a lower case alias for ImageHander.

    :param imagespec: textual image spec
    :param extensions: list of extensions the handler should be applied for
    """
    return MyImageHandler(imagespec, extensions)


def get_sample(url, imagespec="npraw"):
    """
    Get a sample from a URL with basic auto-decoding.
    We can specify the imagespec to decode the image.

    The `imagespec` specifies whether the image is decoded
    to numpy/torch/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - npraw: numpy None None
    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchl: torch float l
    - torchrgb: torch float rgb
    - torch: torch float rgb
    - torchrgba: torch float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba
    """
    if isinstance(url, Path):
        url = url.as_posix()

    img_decoder = my_imagehandler(imagespec)
    dataset = WebDataset(
        url, nodesplitter=non_empty, workersplitter=None, shardshuffle=False
    ).decode(img_decoder, ply_decoder)

    return next(iter(dataset))


############ My custom decoder ############
def ply_decoder(key, data):
    """
    Args:
        key: str, key of the data
        data: bytes, data of the ply file
    Returns:
        plydata: PLYData, ply data
    """
    if not key.endswith(".ply"):
        return None
    from plyfile import PlyData

    with io.BytesIO(data) as byte_stream:
        plydata = PlyData.read(byte_stream)

    return plydata


############ My custom decoder ############


def write_to_tar(sample, output_file, __key__=None):
    if __key__ is not None:
        sample["__key__"] = __key__

    # prepare output file
    if type(output_file) == str:
        output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # write tar file
    sink = wds.TarWriter(str(output_file))
    sink.write(sample)
    sink.close()
    print(f"Saved {output_file}")


def get_sorted_data_in_list(tar_file, must_in_key=[]):
    if isinstance(tar_file, Path):
        tar_file = tar_file.as_posix()

    if isinstance(must_in_key, str):
        must_in_key = [must_in_key]

    data_dict = get_sample(tar_file)
    key_list = [
        key for key in data_dict.keys() if all([must in key for must in must_in_key])
    ]
    key_list.sort()

    data_list = [data_dict[k] for k in key_list]
    return data_list
