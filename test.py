import random
import numpy as np
import torch.random
import os
import einops
import subprocess
import torchio as tio
import json
from tqdm import trange
from data.data import Images3D, mask3d_generator
from pathlib import Path
from options.test_options import TestOptions
from model.net import GMCNN
from model.loss import GeneratorLoss
from util.utils import load_models, process_data, process_generator_out
from monai.networks.nets import UNet


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]
    ))

    config = TestOptions().parse()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    dataset = Images3D(config.data_file, config.root_dir, im_size=config.img_shapes)
    mask_generator = mask3d_generator(config.img_shapes, config.mask_shapes)
    next(mask_generator)

    # generator = GMCNN(in_channels=2, out_channels=1, cnum=config.g_cnum, norm=None).cuda().eval()
    generator = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=[16, 32, 64, 128],
        strides=[2, 2, 2],
    ).cuda().eval()
    load_models(config.load_model_dir, {"generator": generator})

    test_num = len(dataset)
    print(f"Running inference on {test_num} images.")
    
    gen_loss = GeneratorLoss(config.lambda_rec, config.lambda_ae)
    losses = {}
    for i in trange(test_num):
        image = dataset[i]
        mask, rect = next(mask_generator)
        # process_data expects the data to have a batch dimension
        image = einops.repeat(image, 'c d h w -> b c d h w', b=1)
        data = process_data(image, mask, rect)
        predicted = generator(data["gin"])
        gen_out = process_generator_out(predicted, data)
        gen_loss(gen_out, data)

        for ttype, t in zip(("input", "output"), (data["im_in"], gen_out["global"])):
            t = t.squeeze(0).detach().cpu().numpy()
            t_tio = tio.ScalarImage(tensor=t)
            t_tio.save(config.saving_path/f"{ttype}_{i}.nii.gz")

        losses[i] = gen_loss.losses

    with open(config.saving_path/f"loss.json", "w") as f:
        json.dump(losses, f)

    print('done.')