import os
from pathlib import Path

import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import einops
from torchvision import transforms

from options.train_options import TrainOptions
from data.data import Images3D, MedTransform, mask3d_generator
from model.net import InpaintingModel_GMCNN
from util.utils import getLatest

config = TrainOptions().parse()

print('loading data..')
dataset = Images3D(config.data_file, config.root_dir, im_size=config.img_shapes, transform=MedTransform)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
mask_generator = mask3d_generator(config.img_shapes, config.mask_shapes)
next(mask_generator)
print('data loaded..')

print('configuring model..')
ourModel = InpaintingModel_GMCNN(in_channels=2, opt=config).cuda()
ourModel.print_networks()
if config.load_model_dir != '':
    print('Loading pretrained model from {}'.format(config.load_model_dir))
    ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))
    print('Loading done.')

#################################################################################
# #%%
# DEBUGGING SETUP:
# data = next(iter(dataloader))
# data = {k: v.cuda() for k,v in data.items()}
# ourModel.setInput(data)
# # %%
# # need to fix padding
# # issues with resizing (to make life easier change debugConfig to use 128 size)
# ourModel.optimize_parameters()
# ret_losses = ourModel.get_current_losses()
#################################################################################

print('model setting up..')
print('training initializing..')
writer = SummaryWriter(log_dir=config.model_folder)
cnt = 0
for epoch in range(config.epochs):
    for i, data in enumerate(dataloader):
        gt = data.cuda()
        mask, rect = next(mask_generator)
        mask = einops.repeat(mask, 'd h w -> b c d h w', b=config.batch_size, c=1).cuda()
        rect = [rect[0, i] for i in range(6)]
        gt_local = gt[:, :, 
                rect[0]:rect[0] + rect[1],
                rect[2]:rect[2] + rect[3],
                rect[4]:rect[4] + rect[5],]
        im_in = gt * (1 - mask)
        
        # TODO:generate masked inputs
        data_in = {'gt': gt, 'mask': mask, 'rect': rect, "im_in": im_in, "gt_local": gt_local}
        ourModel.setInput(data_in)
        ourModel.optimize_parameters()

        if (i+1) % config.viz_steps == 0:
            ret_loss = ourModel.get_current_losses()
            if config.pretrain_network is False:
                print(
                    '[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f, adv: %.4f), D_loss: %.4f'
                    % (epoch + 1, i + 1, ret_loss['G_loss'], ret_loss['G_loss_rec'], ret_loss['G_loss_ae'],
                       ret_loss['G_loss_adv'], ret_loss['D_loss']))
                writer.add_scalar('adv_loss', ret_loss['G_loss_adv'], cnt)
                writer.add_scalar('D_loss', ret_loss['D_loss'], cnt)
                # writer.add_scalar('G_mrf_loss', ret_loss['G_loss_mrf'], cnt)
            else:
                print('[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f)'
                      % (epoch + 1, i + 1, ret_loss['G_loss'], ret_loss['G_loss_rec'], ret_loss['G_loss_ae']))

            writer.add_scalar('G_loss', ret_loss['G_loss'], cnt)
            writer.add_scalar('reconstruction_loss', ret_loss['G_loss_rec'], cnt)
            writer.add_scalar('autoencoder_loss', ret_loss['G_loss_ae'], cnt)

            # images = ourModel.get_current_visuals_tensor()
            # im_completed = vutils.make_grid(images['completed'], normalize=True, scale_each=True)
            # im_input = vutils.make_grid(images['input'], normalize=True, scale_each=True)
            # im_gt = vutils.make_grid(images['gt'], normalize=True, scale_each=True)
            # writer.add_image('gt', im_gt, cnt)
            # writer.add_image('input', im_input, cnt)
            # writer.add_image('completed', im_completed, cnt)
            if (i+1) % config.train_spe == 0:
                print('saving model ..')
                ourModel.save_networks(epoch+1)
        cnt += 1
    ourModel.save_networks(epoch+1)

writer.export_scalars_to_json(os.path.join(config.model_folder, 'GMCNN_scalars.json'))
writer.close()
