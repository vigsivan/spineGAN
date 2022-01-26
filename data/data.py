from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset

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
        root_dir: Optional[str] = None,
        *,
        im_size: Tuple[int, int, int] = (64, 256, 256),
        transform: tio.Transform = None,
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
            [tio.CropOrPad(im_size), tio.RescaleIntensity(),]
        )

    def __len__(self):
        return len(self.filenames)

    def __resize(self, image: tio.Image) -> tio.Image:
        factor = max(self.im_size) / max(image.spatial_shape)
        new_spacing = tuple([i / factor for i in image.spacing])
        resample = tio.Resample(new_spacing)
        resized = self.processing_tsfm(resample(image))
        return resized

    def __getitem__(self, index) -> torch.tensor:
        image = tio.ScalarImage(self.filenames[index])
        if self.transform:
            image = self.transform(image)
        image = self.__resize(image)
        return image.data



def mask3d_generator(im_size, mask_size, margin=0, ndim=3):
    while True:
        mask = torch.zeros(im_size)
        offsets = [np.random.randint(margin, im_size[i]-mask_size[i]-margin)
                    for i in range(ndim)]
        (o0,o1,o2), (s0, s1, s2) = offsets, mask_size
        mask[o0:o0+s0,o1:o1+s1,o2:o2+s2] = 1
        rect = []
        for oi, si in zip(offsets, mask_size):
            rect.extend([oi, si])
        yield mask, np.array([rect], dtype=int)


class MaskedImages3D(Dataset):
    def __init__(
        self,
        gen: Images3D,
        mask_size: Tuple[int, int, int],
        mask_type: str = "rect",
        mask_margin: int = 0,
    ):

        if mask_type != "rect":
            if mask_type == "stroke":
                raise NotImplementedError("Only implemented rect mask for now!")
            else:
                raise ValueError(f"Mask {mask_type} not supported.")

        self.gen = gen
        self.ndim = 3
        self.im_size = self.gen.im_size
        self.mask_type = mask_type
        self.mask_size = mask_size
        self.margin = mask_margin

    def __len__(self):
        return len(self.gen)

    def __generate_random_rect_mask(self):
        mask = torch.zeros(self.im_size)
        offsets = [
            np.random.randint(
                self.margin, self.im_size[i] - self.mask_size[i] - self.margin
            )
            for i in range(self.ndim)
        ]
        (o0, o1, o2), (s0, s1, s2) = offsets, self.mask_size
        mask[o0 : o0 + s0, o1 : o1 + s1, o2 : o2 + s2] = 1
        rect = []
        for oi, si in zip(offsets, self.mask_size):
            rect.extend([oi, si])
        rect = torch.Tensor([rect])
        return mask, rect

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        image = self.gen[index]
        mask, rect = self.__generate_random_rect_mask()
        masked_image = image * torch.abs(mask - 1)
        return {"gt": image, "masked_image": masked_image, "mask": mask, "rect": rect}


if __name__ == "__main__":
    datafile = "/home/vsivan/t2_list.txt"
    images_gen = Images3D(datafile, transform=MedTransform)
    dataset = MaskedImages3D(images_gen, (32, 128, 128))
    for i in range(len(dataset)):
        dataset[i]

# #%%

# datafile = "/home/vsivan/t2_list.txt"
# images_gen = Images3D(datafile, transform=MedTransform)
# dataset = MaskedImages3D(images_gen, (32,128,128))


# # %%
# for i in range(len(dataset)):
#     data = dataset[i]
# # %%
# dataset[3]
# # %%

