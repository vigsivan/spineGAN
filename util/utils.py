import glob
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import scipy.stats as st
import torch
import einops


def load_models(
    checkpoint_path: Path,
    models: Dict[str, torch.nn.Module],
    epoch: Optional[int] = None,
) -> Path:
    # NOTE: we expect models path files to be saved with the format
    # `model_{epoch_number}.pth' so we use a simple heuristic to
    # load the model from the last-saved epoch (if epoch not provided)

    if epoch is not None:
        assert os.path.exists(checkpoint_path / f"model_{epoch}.pth")
    else:
        epochs = [
            int(f.split(".")[0].split("_")[-1])
            for f in os.listdir(checkpoint_path)
            if f.endswith(".pth")
        ]
        epoch = max(epochs)
    print(f"Loading checkpoint from {epoch=}")

    assert isinstance(epoch, int)
    load_path = checkpoint_path / f"model_{epoch}.pth"
    checkpoint = torch.load(load_path)
    for model_name in models.keys():
        models[model_name].load_state_dict(checkpoint[model_name])

    return load_path


def process_data(data: torch.Tensor, mask: torch.Tensor, rect):
    """Processes the input data into something more convenient"""
    gt = data.cuda()
    batch_size = gt.shape[0]
    mask = einops.repeat(mask, "d h w -> b c d h w", b=batch_size, c=1).cuda()
    rect = [rect[0, i] for i in range(6)]
    gt_local = gt[
        :,
        :,
        rect[0] : rect[0] + rect[1],
        rect[2] : rect[2] + rect[3],
        rect[4] : rect[4] + rect[5],
    ]
    im_in = gt * (1 - mask)
    gin = torch.cat((im_in, mask), dim=1)

    data_in = {
        "gt": gt,
        "mask": mask,
        "rect": rect,
        "im_in": im_in,
        "gt_local": gt_local,
        "gin": gin,
    }
    return data_in

def process_generator_out(
    generator_out: torch.Tensor, inputs: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Gets the global generated image and the local generated image"""
    completed = generator_out * inputs["mask"] + inputs["gt"] * (1 - inputs["mask"])
    completed_local = completed[
        :,
        :,
        inputs["rect"][0] : inputs["rect"][0] + inputs["rect"][1],
        inputs["rect"][2] : inputs["rect"][2] + inputs["rect"][3],
        inputs["rect"][4] : inputs["rect"][4] + inputs["rect"][5],
    ]
    return {"prediction": generator_out, "global": completed, "local": completed_local}


def process_discriminator_out(
    gt_logits: Tuple[torch.Tensor, torch.Tensor],
    generator_logits: Tuple[torch.Tensor, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        "gt_logit": gt_logits[0],
        "gt_logit_local": gt_logits[1],
        "generator_logit": generator_logits[0],
        "generator_logit_local": generator_logits[1],
    }


def gauss_kernel(size=21, sigma=3, inchannels=3, outchannels=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2,sigma+interval/2,size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, size, size))
    out_filter = np.tile(out_filter, [outchannels, inchannels, 1, 1])
    return out_filter


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_rect_mask(im_size, mask_size, margin=8, rand_mask=True):
    mask = np.zeros((im_size[0], im_size[1])).astype(np.float32)
    if rand_mask:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
        of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
    else:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = (im_size[0] - sz0) // 2
        of1 = (im_size[1] - sz1) // 2
    mask[of0:of0+sz0, of1:of1+sz1] = 1
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    rect = np.array([[of0, sz0, of1, sz1]], dtype=int)
    return mask, rect


def generate_stroke_mask(im_size, parts=10, maxVertex=20, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.transpose(mask, [2, 0, 1])
    mask = np.expand_dims(mask, 0)
    return mask


def generate_mask(type, im_size, mask_size):
    if type == 'rect':
        return generate_rect_mask(im_size, mask_size)
    else:
        return generate_stroke_mask(im_size), None


def getLatest(folder_path):
    files = glob.glob(folder_path)
    file_times = list(map(lambda x: time.ctime(os.path.getctime(x)), files))
    return files[sorted(range(len(file_times)), key=lambda x: file_times[x])[-1]]
