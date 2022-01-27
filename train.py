import os

from typing import Generator, Dict
from numpy import pad
from torch.utils.data import DataLoader
from model.basemodel import BaseModel
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import einops

from options.train_options import TrainOptions
from data.data import Images3D, MedTransform, mask3d_generator
from model.net import InpaintingModel_GMCNN
from util.utils import getLatest

def log_losses(
    losses: Dict[str, float],
    writer: SummaryWriter,
    *,
    epoch,
    step,
    total_steps,
    pretrain_network: bool
):
    if pretrain_network is False:
        print(
            "[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f, adv: %.4f), D_loss: %.4f"
            % (
                epoch + 1,
                step + 1,
                losses["G_loss"],
                losses["G_loss_rec"],
                losses["G_loss_ae"],
                losses["G_loss_adv"],
                losses["D_loss"],
            )
        )
        writer.add_scalar("adv_loss", losses["G_loss_adv"], total_steps)
        writer.add_scalar("D_loss", losses["D_loss"], total_steps)
        # writer.add_scalar('G_mrf_loss', losses['G_loss_mrf'], total_steps)
    else:
        print(
            "[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f)"
            % (
                epoch + 1,
                step + 1,
                losses["G_loss"],
                losses["G_loss_rec"],
                losses["G_loss_ae"],
            )
        )

    writer.add_scalar("G_loss", losses["G_loss"], total_steps)
    writer.add_scalar("reconstruction_loss", losses["G_loss_rec"], total_steps)
    writer.add_scalar("autoencoder_loss", losses["G_loss_ae"], total_steps)

def init_optimizers(model: InpaintingModel_GMCNN, lr: float, pretrain_network: bool):
    optimizer_G = torch.optim.Adam(
        model.netGM.parameters(), lr=lr, betas=(0.5, 0.9)
    )
    if not pretrain_network:
        optimizer_D = torch.optim.Adam(
                    filter(lambda x: x.requires_grad, model.netD.parameters()),
                    lr=lr,
                    betas=(0.5, 0.9),
        )
    else:
        optimizer_D = None
    return {
        "optimizer_G": optimizer_G,
        "optimizer_D": optimizer_D
    }

def training_loop(
    model: InpaintingModel_GMCNN,
    optimizers: Dict[str, torch.optim.Optimizer],
    dataloader: DataLoader,
    mask_generator: Generator,
    pretrain_network: bool,
    batch_size: int,
    epochs: int,
    spe: int,
    writer: SummaryWriter,
    starting_epoch: int = 0,
    viz_steps: int = 5,
):
    erange = range(starting_epoch, starting_epoch + epochs)
    steps = 0
    for epoch in erange:
        for i, data in enumerate(dataloader):
            gt = data.cuda()
            mask, rect = next(mask_generator)
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

            data_in = {
                "gt": gt,
                "mask": mask,
                "rect": rect,
                "im_in": im_in,
                "gt_local": gt_local,
                **optimizers
            }
            model.setInput(data_in)
            model.optimize_parameters()

            if (i + 1) % viz_steps == 0:
                ret_loss = model.get_current_losses()
                log_losses(
                    ret_loss,
                    writer,
                    epoch=epoch,
                    step=i,
                    total_steps=steps,
                    pretrain_network=pretrain_network,
                )

                if (i + 1) % spe == 0:
                    print("saving model ..")
                    model.save_networks(epoch + 1)
            steps += 1


def train(
    config,
    dataloader: DataLoader,
    mask_generator: Generator,
    writer: SummaryWriter,
    *,
    pretrain: bool = True
):
    print("configuring model..")
    ourModel = InpaintingModel_GMCNN(in_channels=2, opt=config).cuda()
    optimizers = init_optimizers(ourModel, config.lr, pretrain_network=pretrain)
    ourModel.print_networks()
    if config.load_model_dir != "":
        print("Loading pretrained model from {}".format(config.load_model_dir))
        ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, "*.pth")))
        print("Loading done.")

    
    print("model setting up..")
    print("training initializing..")
    training_args = {
        "model": ourModel,
        "optimizers": optimizers,
        "dataloader": dataloader,
        "mask_generator": mask_generator,
        "pretrain_network": pretrain,
        "batch_size": config.batch_size,
        "epochs": config.pretrain_epochs if pretrain else config.finetune_epochs,
        "spe": config.train_spe,
        "writer": writer,
        "starting_epoch": 0 if pretrain else config.pretrain_epochs + 1,
        "viz_steps": 5,
    }
    training_loop(**training_args)
    total_epochs = config.pretrain_epochs if pretrain else config.finetune_epochs + config.pretrain_epochs 
    ourModel.save_networks(total_epochs)


if __name__ == "__main__":
    config = TrainOptions().parse()
    print("loading data..")
    dataset = Images3D(
        config.data_file,
        config.root_dir,
        im_size=config.img_shapes,
        transform=MedTransform,
        pad_mode=config.pad_mode
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    mask_generator = mask3d_generator(config.img_shapes, config.mask_shapes)
    next(mask_generator)
    print("data loaded..")

    writer = SummaryWriter(log_dir=config.model_folder)

    print("Pretraining network...")
    train(config, dataloader, mask_generator, writer, pretrain=True)

    config.pretrain_network = False
    config.load_model_dir = config.model_folder

    print("Pretraining finetuning...")
    train(config, dataloader, mask_generator, writer, pretrain=False)

    writer.export_scalars_to_json(
        os.path.join(config.model_folder, "GMCNN_scalars.json")
    )
    writer.close()
