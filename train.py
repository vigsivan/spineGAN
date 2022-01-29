import os
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data.data import Images3D, MedTransform, mask3d_generator
from model.loss import DiscriminatorLoss, GeneratorLoss, ModelLoss
from model.net import GMCNN, GlobalLocalDiscriminator  # InpaintingModel_GMCNN
from options.train_options import TrainOptions


def log_losses(
    losses: Dict[str, float],
    writer: SummaryWriter,
    *,
    epoch,
    step,
    total_steps,
    print_to_console: bool = False,
):
    for loss_name, loss in losses.items():
        writer.add_scalar(loss_name, loss, total_steps)

    if print_to_console:
        loss_str = f"[{epoch}  {step}] "
        loss_str += "[" + " ".join([f"({k}: {v})" for k, v in losses.items()]) + "]"
        print(loss_str)


def process_data(data, mask, rect):
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


def get_models_optimizers_and_losses(
    config, act=F.elu, ndim: int = 3
) -> Tuple[
    Dict[str, torch.nn.Module], Dict[str, torch.optim.Optimizer], Dict[str, ModelLoss]
]:
    g_fc_channels = (np.prod(config.img_shapes) * config.d_cnum * 4) // (16 ** ndim)
    l_fc_channels = (np.prod(config.mask_shapes) * config.d_cnum * 4) // (16 ** ndim)

    generator = GMCNN(
        in_channels=2, out_channels=1, cnum=config.g_cnum, act=act, norm=None
    ).cuda()
    discriminator = GlobalLocalDiscriminator(
        1,
        cnum=config.d_cnum,
        act=act,
        g_fc_channels=g_fc_channels,
        l_fc_channels=l_fc_channels,
        spectral_norm=config.spectral_norm,
    ).cuda()

    models = {"generator": generator, "discriminator": discriminator}

    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.lr, betas=(0.5, 0.9)
    )
    d_optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, discriminator.parameters()),
        lr=config.lr,
        betas=(0.5, 0.9),
    )

    optimizers = {"generator": g_optimizer, "discriminator": d_optimizer}

    if config.pretrain_network:
        g_loss = GeneratorLoss(config.lambda_rec, config.lambda_ae)
    else:
        g_loss = GeneratorLoss(
            config.lambda_rec, config.lambda_ae, lambda_adversarial=config.lambda_adv
        )

    g_loss = g_loss.cuda()
    d_loss = DiscriminatorLoss().cuda()

    losses = {"generator": g_loss, "discriminator": d_loss}

    return models, optimizers, losses


def save_models(models: Dict[str, torch.nn.Module], checkpoint_path: Path, epoch: int):
    torch.save(
        {
            "generator": models["generator"].state_dict(),
            "discriminator": models["discriminator"].state_dict(),
        },
        Path(checkpoint_path) / f"model_{epoch}.pth",
    )


def training_loop(
    models: Dict[str, torch.nn.Module],
    losses: Dict[str, ModelLoss],
    optimizers: Dict[str, torch.optim.Optimizer],
    dataloader: DataLoader,
    mask_generator: Generator,
    pretrain_network: bool,
    discriminator_iters: int,
    checkpoint_path: Path,
    epochs: int,
    saves_per_epoch: int,
    writer: SummaryWriter,
    starting_epoch: int = 0,
    viz_steps: int = 5,
):
    erange = range(starting_epoch, starting_epoch + epochs)
    steps = 0
    for epoch in erange:
        for i, data in enumerate(dataloader):
            mask, rect = next(mask_generator)
            inputs = process_data(data, mask, rect)
            gen_out = models["generator"](inputs["gin"])
            gen_out = process_generator_out(gen_out, inputs)

            disc_out = None

            if not pretrain_network:
                for _ in range(discriminator_iters):
                    optimizers["discriminator"].zero_grad()
                    gt_logits = models["discriminator"](
                        inputs["gt"].detach(), inputs["gt_local"].detach()
                    )
                    generated_logits = models["discriminator"](
                        gen_out["global"].detach(), gen_out["local"].detach()
                    )
                    disc_out = process_discriminator_out(gt_logits, generated_logits)
                    d_loss = losses["discriminator"](disc_out)
                    d_loss.backward(retain_graph=True)
                    optimizers["discriminator"].step()

                # NOTE: we use the outputs of the discriminator to do another forward pass
                # so we recompute
                gt_logits = models["discriminator"](
                    inputs["gt"].detach(), inputs["gt_local"].detach()
                )
                generated_logits = models["discriminator"](
                    gen_out["global"].detach(), gen_out["local"].detach()
                )
                disc_out = process_discriminator_out(gt_logits, generated_logits)

            optimizers["generator"].zero_grad()
            g_loss = losses["generator"](gen_out, inputs, disc_out)
            g_loss.backward()
            optimizers["generator"].step()

            if (i + 1) % viz_steps == 0:
                ret_loss = {
                    **losses["generator"].get_losses(),
                    **losses["discriminator"].get_losses(),
                }
                log_losses(
                    losses=ret_loss,
                    writer=writer,
                    epoch=epoch,
                    step=i,
                    total_steps=steps,
                    print_to_console=True,
                )
            steps += 1

        if epoch % saves_per_epoch == 0:
            print("saving model ..")
            save_models(models, checkpoint_path, epoch)


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

    assert isinstance(epoch, int)
    load_path = checkpoint_path / f"model_{epoch}.pth"
    checkpoint = torch.load(load_path)
    for model_name in models.keys():
        models[model_name].load_state_dict(checkpoint[model_name])

    return load_path


def train(
    config, dataloader: DataLoader, mask_generator: Generator, writer: SummaryWriter,
):
    pretrain = config.pretrain_network
    print("configuring models..")
    models, optimizers, losses = get_models_optimizers_and_losses(config)

    if config.load_model_dir:
        print("Loading pretrained model from {}".format(config.load_model_dir))
        load_path = load_models(config.load_model_dir, models)
        print(f"Loaded model file {load_path}.")

    print("model setting up..")
    print("training initializing..")
    training_args = {
        "models": models,
        "losses": losses,
        "optimizers": optimizers,
        "dataloader": dataloader,
        "mask_generator": mask_generator,
        "pretrain_network": pretrain,
        "discriminator_iters": config.D_max_iters,
        # "batch_size": config.batch_size,
        "checkpoint_path": config.checkpoint_dir,
        "epochs": config.pretrain_epochs if pretrain else config.finetune_epochs,
        "saves_per_epoch": config.train_spe,
        "writer": writer,
        "starting_epoch": 0 if pretrain else config.pretrain_epochs + 1,
        "viz_steps": 5,
    }
    training_loop(**training_args)
    total_epochs = (
        config.pretrain_epochs
        if pretrain
        else config.finetune_epochs + config.pretrain_epochs
    )
    save_models(models, config.checkpoint_dir, total_epochs)
    # ourModel.save_networks(total_epochs)


if __name__ == "__main__":
    config = TrainOptions().parse()
    print("loading data..")
    dataset = Images3D(
        config.data_file,
        config.root_dir,
        im_size=config.img_shapes,
        transform=MedTransform,
        pad_mode=config.pad_mode,
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    mask_generator = mask3d_generator(config.img_shapes, config.mask_shapes)
    next(mask_generator)
    print("data loaded..")

    writer = SummaryWriter(log_dir=config.checkpoint_dir)

    print("Pretraining network...")
    train(config, dataloader, mask_generator, writer)

    config.pretrain_network = False
    config.load_model_dir = config.checkpoint_dir

    print("Finetuning network...")
    train(config, dataloader, mask_generator, writer)

    writer.export_scalars_to_json(
        os.path.join(config.model_folder, "GMCNN_scalars.json")
    )
    writer.close()
