import random
import numpy as np
import torch.random
import os
import einops
import subprocess
import torchio as tio
from tqdm import trange
from data.data import Images3D, mask3d_generator
from pathlib import Path
from options.test_options import TestOptions
from model.net import InpaintingModel_GMCNN, generate_mask3d
from util.utils import generate_rect_mask, generate_stroke_mask, getLatest

os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]
        ))


config = TestOptions().parse()
if config.random_mask:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.random.seed(config.seed)

dataset = Images3D(config.data_file, config.root_dir, im_size=config.img_shapes)
mask_generator = mask3d_generator(config.img_shapes, config.mask_shapes)
next(mask_generator)

print('configuring model..')
ourModel = InpaintingModel_GMCNN(in_channels=2, opt=config)
ourModel.print_networks()
ourModel.cuda()

if not config.scratch:
    print('Loading pretrained model from {}'.format(config.load_model_dir))
    ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))
    print('Loading done.')
else:
    print("Testing from scratch")

test_num = len(dataset)
print(f"Running inference on {test_num} images.")

for i in trange(test_num):
    image = dataset[i]
    mask, _ = next(mask_generator)

    image = einops.repeat(image, 'c d h w -> b c d h w', b=1).cuda()
    mask = einops.repeat(mask, 'd h w -> b c d h w', b=1, c=1).cuda()
    im_in = image * (1 - mask)
    
    result = ourModel.evaluate(im_in, mask).detach().cpu()
    input_tio = tio.ScalarImage(tensor=im_in.detach().cpu().squeeze().unsqueeze(0))
    input_tio.save(config.saving_path/f"input_{i}.nii.gz")
    
    # result = einops.rearrange(result, 'b c d h w -> (b c) d fh w')
    result_tio = tio.ScalarImage(tensor=result.squeeze().unsqueeze(0))
    result_tio = tio.RescaleIntensity()(result_tio)
    result_tio.save(config.saving_path/f"output_{i}.nii.gz") 
print('done.')
