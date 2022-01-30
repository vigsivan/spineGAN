from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import cv2
import torch
import torchio as tio
from torch.utils.data import Dataset, DataLoader

MedTransform: tio.Transform = tio.Compose(
    [
        tio.RandomAnisotropy(),
        tio.RandomAffine(),
        tio.RandomGhosting(),
        tio.RandomNoise(),
        tio.RandomBlur(),
    ]
)


class Images3D(Dataset):
    def __init__(
        self,
        files_list: Union[Path, str],
        root_dir: Optional[Path] = None,
        *,
        im_size: Tuple[int, int, int] = (64, 256, 256),
        transform: tio.Transform = None,
        return_tio: bool = False,
        pad_mode: Optional[str] = None,
        int_min_max: Tuple[int, int] = (-1, 1)
    ):
        with open(files_list, "rt") as f:
            filenames = f.read().splitlines()
        if root_dir:
            self.filenames = [Path(root_dir) / Path(f) for f in filenames]
        else:
            self.filenames = [Path(f) for f in filenames]

        self.transform = transform
        self.im_size = im_size
        pad_mode = pad_mode if pad_mode else 0.0
        self.processing_tsfm = tio.Compose(
            [
                tio.CropOrPad(im_size, padding_mode=pad_mode),
                tio.RescaleIntensity(out_min_max=int_min_max),
            ]
        )
        self.return_tio = return_tio

    def __len__(self):
        return len(self.filenames)

    def __resize(self, image: tio.Image) -> tio.Image:
        factor = max(self.im_size) / max(image.spatial_shape)
        new_spacing = tuple([i / factor for i in image.spacing])
        resample = tio.Resample(new_spacing)
        resized = self.processing_tsfm(resample(image))
        return resized

    def __getitem__(self, index) -> Union[torch.tensor, tio.ScalarImage]:
        image = tio.ScalarImage(self.filenames[index])
        if self.transform:
            image = self.transform(image)
        image = self.__resize(image)
        return image if self.return_tio else image.data

class InpaintingDataset(Dataset):
    def __init__(self, info_list, root_dir='', im_size=(256, 256), transform=None):
        self.filenames = open(info_list, 'rt').read().splitlines()
        self.root_dir = root_dir
        self.transform = transform
        self.im_size = im_size
        np.random.seed(2018)

    def __len__(self):
        return len(self.filenames)

    def read_image(self, filepath):
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            h, w, _ = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1],:]
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
        else:
            im_scaled = np.transpose(image, [2, 0, 1])
        return im_scaled

    def __getitem__(self, idx):
        image = self.read_image(os.path.join(self.root_dir, self.filenames[idx]))
        sample = {'gt': image}
        if self.transform:
            sample = self.transform(sample)
        return sample


def mask3d_generator(
    im_size, mask_size, margin=0, device: torch.device = torch.device("cuda")
):
    ndim = 3
    while True:
        mask = torch.zeros(im_size, device=device)
        offsets = [
            np.random.randint(margin, im_size[i] - mask_size[i] - margin)
            for i in range(ndim)
        ]
        (o0, o1, o2), (s0, s1, s2) = offsets, mask_size
        mask[o0 : o0 + s0, o1 : o1 + s1, o2 : o2 + s2] = 1
        rect = []
        for oi, si in zip(offsets, mask_size):
            rect.extend([oi, si])
        yield mask, np.array([rect], dtype=int)


#%%
root = Path("/media/vsivan/Untitled/spine-generic-multisubject/t2/")
files_list = Path("/home/vsivan/gan_experiments/train_list.txt")
im_size = (32, 128, 128)
dataset = Images3D(files_list, root, im_size=im_size, transform=MedTransform)

#%%


if __name__ == "__main__":
    import os
    import typer
    import einops
    from tqdm import trange
    import matplotlib.pyplot as plt

    app = typer.Typer()

    @app.command()
    def save_dataset_files(
        files_list: Path,
        root_dir: Path,
        savedir: Path,
        savenum: int = 10,
        im_size: Tuple[int, int, int] = (64, 256, 256),
        mask_size: Optional[Tuple[int, int, int]] = None,
    ):
        if not mask_size:
            mask_size = tuple([i // 2 for i in im_size])

        os.makedirs(savedir, exist_ok=True)

        dataset = Images3D(
            files_list, root_dir, im_size=im_size, transform=MedTransform
        )
        mask_gen = mask3d_generator(im_size, mask_size)
        next(mask_gen)
        for i in trange(savenum):
            image = dataset[i]
            mask, _ = next(mask_gen)
            mask = einops.repeat(mask, "d h w -> c d h w", c=1)
            im_in = image * (1 - mask)
            im_in_tio = tio.ScalarImage(tensor=im_in)
            im_in_tio.save(savedir / f"input_{i+10}.nii.gz")

    @app.command()
    def print_resampled_shapes(
        files_list: Path, root_dir: Path, im_size: Tuple[int, int, int]
    ):
        with open(files_list, "rt") as f:
            filenames = f.read().splitlines()
            filenames = [root_dir / f for f in filenames]
            for f in filenames:
                image = tio.ScalarImage(f)
                factor = max(im_size) / max(image.spatial_shape)
                new_spacing = tuple([i / factor for i in image.spacing])
                resample = tio.Resample(new_spacing)
                resampled = resample(image)
                print(f"{f}: {resampled.spatial_shape}")

    @app.command()
    def create_static_slides_dataset(
        files_list: Path, root_dir: Path, save_dir: Path, im_size: Tuple[int, int, int],
    ):
        dataset = Images3D(
            files_list, root_dir, im_size=im_size, transform=MedTransform, int_min_max=(0, 1)
        )
        with open(files_list, "r") as f:
            file_names = f.readlines()
            file_names = [fn.strip().split(".")[0] for fn in file_names]

        os.makedirs(save_dir, exist_ok=True)

        mid_slice = im_size[0] // 2
        slices = [i for i in range(mid_slice - 1, mid_slice + 2)]
        for i in trange(len(dataset)):
            data = dataset[i]  # c, d, h, w
            slice_tensors = [data[0, sli, :, :] for sli in slices]
            im_tensors = [
                einops.repeat(st, "h w -> h w c", c=3) for st in slice_tensors
            ]
            for im_tensor in im_tensors:
                im_np = im_tensor.numpy()
                plt.imsave(save_dir / f"{file_names[i]}.png", im_np)

    app()
# %%
