from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
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
    ):
        with open(files_list, "rt") as f:
            filenames = f.read().splitlines()
        if root_dir:
            self.filenames = [Path(root_dir) / Path(f) for f in filenames]
        else:
            self.filenames = [Path(f) for f in filenames]

        self.transform = transform
        self.im_size = im_size
        self.processing_tsfm = tio.Compose(
            [tio.CropOrPad(im_size), tio.RescaleIntensity(out_min_max=(-1,1)),]
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


def mask3d_generator(im_size, mask_size, margin=0):
    ndim = 3
    while True:
        mask = torch.zeros(im_size)
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


if __name__ == "__main__":
    import os
    import typer
    import einops
    from tqdm import trange

    app = typer.Typer()

    @app.command()
    def main(
        files_list: Path,
        root_dir: Path,
        savedir: Path,
        savenum: int=10,
        im_size: Tuple[int, int, int] = (64, 256, 256),
        mask_size: Optional[Tuple[int, int, int]] = None,
    ):
        if not mask_size:
            mask_size = tuple([i//2 for i in im_size])

        os.makedirs(savedir, exist_ok=True)
    
        dataset = Images3D(files_list, root_dir, im_size=im_size, transform=MedTransform)
        mask_gen = mask3d_generator(im_size, mask_size)
        next(mask_gen)
        for i in trange(savenum):
            image = dataset[i]
            mask, _ = next(mask_gen)
            mask = einops.repeat(mask, 'd h w -> c d h w', c=1)
            im_in = image * (1 - mask)
            im_in_tio = tio.ScalarImage(tensor=im_in)
            im_in_tio.save(savedir/f"input_{i+10}.nii.gz")

    app()