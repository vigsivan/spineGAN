import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UpSample
from model.basemodel import BaseModel
from model.basenet import BaseNet
from model.loss import WGANLoss, IDMRFLoss, StyleLoss
from model.layer import (
    init_weights,
    PureUpsampling,
    ConfidenceDrivenMaskLayer3D,
    SpectralNorm,
)
import numpy as np

# generative multi-column convolutional neural net
class GMCNN(BaseNet):
    def __init__(
        self,
        in_channels,
        out_channels,
        cnum=32,
        act=F.elu,
        norm=F.instance_norm,
        using_norm=False,
    ):
        super(GMCNN, self).__init__()
        self.act = act
        self.using_norm = using_norm
        if using_norm is True:
            self.norm = norm
        else:
            self.norm = None
        ch = cnum

        # network structure
        self.EB1 = []
        self.EB2 = []
        self.EB3 = []
        self.decoding_layers = []

        self.EB1_pad_rec = []
        self.EB2_pad_rec = []
        self.EB3_pad_rec = []

        self.EB1.append(nn.Conv3d(in_channels, ch, kernel_size=7, stride=1))

        self.EB1.append(nn.Conv3d(ch, ch * 2, kernel_size=7, stride=2))
        self.EB1.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=7, stride=1))

        self.EB1.append(nn.Conv3d(ch * 2, ch * 4, kernel_size=7, stride=2))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1))

        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=2))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=4))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=8))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=16))

        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1))

        self.EB1.append(
            UpSample(
                spatial_dims=3, in_channels=ch * 4, out_channels=ch * 4, scale_factor=4
            )
        )
        # self.EB1.append(PureUpsampling(scale=4))

        self.EB1_pad_rec = [3, 3, 3, 3, 3, 3, 6, 12, 24, 48, 3, 3, 0]

        self.EB2.append(nn.Conv3d(in_channels, ch, kernel_size=5, stride=1))

        self.EB2.append(nn.Conv3d(ch, ch * 2, kernel_size=5, stride=2))
        self.EB2.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=5, stride=1))

        self.EB2.append(nn.Conv3d(ch * 2, ch * 4, kernel_size=5, stride=2))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1))

        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=2))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=4))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=8))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=16))

        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1))

        self.EB2.append(
            UpSample(
                spatial_dims=3, in_channels=ch * 4, out_channels=ch * 4, scale_factor=2
            )
        )
        self.EB2.append(nn.Conv3d(ch * 4, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(
            UpSample(
                spatial_dims=3, in_channels=ch * 2, out_channels=ch * 2, scale_factor=2
            )
        )
        self.EB2_pad_rec = [2, 2, 2, 2, 2, 2, 4, 8, 16, 32, 2, 2, 0, 2, 2, 0]

        self.EB3.append(nn.Conv3d(in_channels, ch, kernel_size=3, stride=1))

        self.EB3.append(nn.Conv3d(ch, ch * 2, kernel_size=3, stride=2))
        self.EB3.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=3, stride=1))

        self.EB3.append(nn.Conv3d(ch * 2, ch * 4, kernel_size=3, stride=2))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1))

        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=2))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=4))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=8))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=16))

        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1))

        self.EB3.append(
            UpSample(
                spatial_dims=3, in_channels=ch * 4, out_channels=ch * 4, scale_factor=2
            )
        )
        self.EB3.append(nn.Conv3d(ch * 4, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(
            UpSample(
                spatial_dims=3, in_channels=ch * 2, out_channels=ch * 2, scale_factor=2
            )
        )
        self.EB3.append(nn.Conv3d(ch * 2, ch, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv3d(ch, ch, kernel_size=3, stride=1))

        self.EB3_pad_rec = [1, 1, 1, 1, 1, 1, 2, 4, 8, 16, 1, 1, 0, 1, 1, 0, 1, 1]

        self.decoding_layers.append(nn.Conv3d(ch * 7, ch // 2, kernel_size=3, stride=1))
        self.decoding_layers.append(
            nn.Conv3d(ch // 2, out_channels, kernel_size=3, stride=1)
        )

        self.decoding_pad_rec = [1, 1]

        self.EB1 = nn.ModuleList(self.EB1)
        self.EB2 = nn.ModuleList(self.EB2)
        self.EB3 = nn.ModuleList(self.EB3)
        self.decoding_layers = nn.ModuleList(self.decoding_layers)

        required_padding = list(
            set(
                self.EB1_pad_rec
                + self.EB2_pad_rec
                + self.EB3_pad_rec
                + self.decoding_pad_rec
            )
        )
        self.pad_list = [nn.ReflectionPad3d(i) for i in required_padding]
        self.pads = {pad: self.pad_list[i] for i, pad in enumerate(required_padding)}

        # # padding operations
        # padlen = 49
        # self.pads = [0] * padlen
        # for i in range(padlen):
        #     self.pads[i] = nn.ReflectionPad3d(i)
        # self.pads = nn.ModuleList(self.pads)

        self.pad_list = nn.ModuleList(self.pad_list)
        # self.pads = lambda p: self.pad_list[self.pad_map[p]]

    def __pad_hack(self, x, pad_idx):
        if pad_idx >= x.shape[2]:
            f = 1
            while pad_idx // f >= x.shape[2]:
                f *= 2
            pad_idx = pad_idx // f
            for _ in range(f - 1):
                x = self.pads[pad_idx](x)
        return self.pads[pad_idx](x)

    def forward(self, x):
        x1, x2, x3 = x, x, x

        for i, layer in enumerate(self.EB1):
            pad_idx = self.EB1_pad_rec[i]
            x1 = self.__pad_hack(x1, pad_idx)
            x1 = layer(x1)
            if self.using_norm:
                x1 = self.norm(x1)
            if pad_idx != 0:
                x1 = self.act(x1)

        for i, layer in enumerate(self.EB2):
            pad_idx = self.EB2_pad_rec[i]
            x2 = self.__pad_hack(x2, pad_idx)
            x2 = layer(x2)
            if self.using_norm:
                x2 = self.norm(x2)
            if pad_idx != 0:
                x2 = self.act(x2)

        for i, layer in enumerate(self.EB3):
            pad_idx = self.EB3_pad_rec[i]
            x3 = self.__pad_hack(x3, pad_idx)
            x3 = layer(x3)
            if self.using_norm:
                x3 = self.norm(x3)
            if pad_idx != 0:
                x3 = self.act(x3)

        x_d = torch.cat((x1, x2, x3), 1)
        x_d = self.act(
            self.decoding_layers[0](self.pads[self.decoding_pad_rec[0]](x_d))
        )
        x_d = self.decoding_layers[1](self.pads[self.decoding_pad_rec[1]](x_d))
        x_out = torch.clamp(x_d, -1, 1) # NOTE: different dynamic range?
        return x_out


# return one dimensional output indicating the probability of realness or fakeness
class Discriminator(BaseNet):
    def __init__(
        self,
        in_channels,
        cnum=32,
        fc_channels=8 * 8 * 32 * 4,
        act=F.elu,
        norm=None,
        spectral_norm=True,
    ):
        super(Discriminator, self).__init__()
        self.act = act
        self.norm = norm
        # self.embedding = None
        # self.logit = None

        ch = cnum
        self.layers = []
        if spectral_norm:
            self.layers.append(
                SpectralNorm(
                    nn.Conv3d(in_channels, ch, kernel_size=5, padding=2, stride=2)
                )
            )
            self.layers.append(
                SpectralNorm(nn.Conv3d(ch, ch * 2, kernel_size=5, padding=2, stride=2))
            )
            self.layers.append(
                SpectralNorm(
                    nn.Conv3d(ch * 2, ch * 4, kernel_size=5, padding=2, stride=2)
                )
            )
            self.layers.append(
                SpectralNorm(
                    nn.Conv3d(ch * 4, ch * 4, kernel_size=5, padding=2, stride=2)
                )
            )
            self.layers.append(SpectralNorm(nn.Linear(fc_channels, 1)))
        else:
            self.layers.append(
                nn.Conv3d(in_channels, ch, kernel_size=5, padding=2, stride=2)
            )
            self.layers.append(
                nn.Conv3d(ch, ch * 2, kernel_size=5, padding=2, stride=2)
            )
            self.layers.append(
                nn.Conv3d(ch * 2, ch * 4, kernel_size=5, padding=2, stride=2)
            )
            self.layers.append(
                nn.Conv3d(ch * 4, ch * 4, kernel_size=5, padding=2, stride=2)
            )
            self.layers.append(nn.Linear(fc_channels, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for _, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.norm is not None:
                x = self.norm(x)
            x = self.act(x)
        embedding = x.view(x.size(0), -1)
        logit = self.layers[-1](embedding)
        return logit


class GlobalLocalDiscriminator(BaseNet):
    def __init__(
        self,
        in_channels,
        cnum=32,
        g_fc_channels=16 * 16 * 32 * 4,
        l_fc_channels=8 * 8 * 32 * 4,
        act=F.elu,
        norm=None,
        spectral_norm=True,
    ):
        super(GlobalLocalDiscriminator, self).__init__()
        self.act = act
        self.norm = norm

        self.global_discriminator = Discriminator(
            in_channels=in_channels,
            fc_channels=g_fc_channels,
            cnum=cnum,
            act=act,
            norm=norm,
            spectral_norm=spectral_norm,
        )
        self.local_discriminator = Discriminator(
            in_channels=in_channels,
            fc_channels=l_fc_channels,
            cnum=cnum,
            act=act,
            norm=norm,
            spectral_norm=spectral_norm,
        )

    def forward(self, x_g, x_l):
        x_global = self.global_discriminator(x_g)
        x_local = self.local_discriminator(x_l)
        return x_global, x_local


def generate_mask3d(im_size, mask_size, margin=0, ndim=3):
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
    return mask, np.array([rect], dtype=int)


class InpaintingModel_GMCNN(BaseModel):
    def __init__(self, in_channels, act=F.elu, norm=None, opt=None):
        super(InpaintingModel_GMCNN, self).__init__()
        self.opt = opt
        self.init(opt)

        self.confidence_mask_layer = ConfidenceDrivenMaskLayer3D()

        self.netGM = GMCNN(
            in_channels, out_channels=1, cnum=opt.g_cnum, act=act, norm=norm
        ).cuda()
        init_weights(self.netGM)
        self.model_names = ["GM"]
        if self.opt.phase == "test":
            return

        self.netD = None

        # self.optimizer_G = torch.optim.Adam(
        #     self.netGM.parameters(), lr=opt.lr, betas=(0.5, 0.9)
        # )
        self.optimizer_G = None
        self.optimizer_D = None

        self.wganloss = None
        self.recloss = nn.L1Loss()
        self.aeloss = nn.L1Loss()
        self.mrfloss = None
        self.lambda_adv = opt.lambda_adv
        self.lambda_rec = opt.lambda_rec
        self.lambda_ae = opt.lambda_ae
        self.lambda_gp = opt.lambda_gp
        self.lambda_mrf = opt.lambda_mrf
        self.G_loss = None
        self.G_loss_reconstruction = None
        self.G_loss_mrf = None
        self.G_loss_adv, self.G_loss_adv_local = None, None
        self.G_loss_ae = None
        self.D_loss, self.D_loss_local = None, None
        self.GAN_loss = None

        self.gt, self.gt_local = None, None
        self.mask, self.mask_01 = None, None
        self.rect = None
        self.im_in, self.gin = None, None

        self.completed, self.completed_local = None, None
        self.completed_logit, self.completed_local_logit = None, None
        self.gt_logit, self.gt_local_logit = None, None

        self.pred = None

        if self.opt.pretrain_network is False:
            ndim = len(opt.img_shapes)
            g_fc_channels = (np.prod(opt.img_shapes) * opt.d_cnum * 4) // (16 ** ndim)
            l_fc_channels = (np.prod(opt.mask_shapes) * opt.d_cnum * 4) // (16 ** ndim)
            if self.opt.mask_type == "rect":
                self.netD = GlobalLocalDiscriminator(
                    1,
                    cnum=opt.d_cnum,
                    act=act,
                    g_fc_channels=g_fc_channels,
                    l_fc_channels=l_fc_channels,
                    spectral_norm=self.opt.spectral_norm,
                ).cuda()
            else:
                self.netD = GlobalLocalDiscriminator(
                    1,
                    cnum=opt.d_cnum,
                    act=act,
                    spectral_norm=self.opt.spectral_norm,
                    g_fc_channels=g_fc_channels,
                    l_fc_channels=l_fc_channels,
                ).cuda()
            init_weights(self.netD)
            # self.optimizer_D = torch.optim.Adam(
            #     filter(lambda x: x.requires_grad, self.netD.parameters()),
            #     lr=opt.lr,
            #     betas=(0.5, 0.9),
            # )
            self.wganloss = WGANLoss()
            self.mrfloss = IDMRFLoss()

    def initVariables(self):
        self.gt, self.mask_01, self.rect, self.im_in, self.gt_local = (
            self.input["gt"],
            self.input["mask"],
            self.input["rect"],
            self.input["im_in"],
            self.input["gt_local"]
        )
        self.optimizer_G, self.optimizer_D = (
            self.input["optimizer_G"],
            self.input["optimizer_D"]
        )
        self.mask = self.confidence_mask_layer(self.mask_01)
        self.gin = torch.cat((self.im_in, self.mask_01), dim=1)

    def forward_G(self):
        self.G_loss_reconstruction = self.recloss(
            self.completed * self.mask, self.gt.detach() * self.mask
        )
        self.G_loss_reconstruction = self.G_loss_reconstruction / torch.mean(
            self.mask_01
        )
        self.G_loss_ae = self.aeloss(
            self.pred * (1 - self.mask_01), self.gt.detach() * (1 - self.mask_01)
        )
        self.G_loss_ae = self.G_loss_ae / torch.mean(1 - self.mask_01)
        self.G_loss = (
            self.lambda_rec * self.G_loss_reconstruction
            + self.lambda_ae * self.G_loss_ae
        )
        if self.opt.pretrain_network is False:
            # discriminator
            self.completed_logit, self.completed_local_logit = self.netD(
                self.completed, self.completed_local
            )
            # generate 2D image slices
            scanning_dim = self.completed_local.shape[2]
            pred_slice = self.completed_local[:, :, scanning_dim // 2, :, :]
            gt_slice = self.gt_local.detach()[:, :, scanning_dim // 2, :, :]
            pred_slice = einops.repeat(
                pred_slice, "b c h w -> b (repeat c) h w", repeat=3
            )
            gt_slice = einops.repeat(gt_slice, "b c h w -> b (repeat c) h w", repeat=3)

            self.G_loss_mrf = self.mrfloss(pred_slice, gt_slice)

            # self.G_loss_mrf = self.mrfloss((self.completed_local+1)/2.0, (self.gt_local.detach()+1)/2.0)
            self.G_loss = self.G_loss + self.lambda_mrf * self.G_loss_mrf

            self.G_loss_adv = -self.completed_logit.mean()
            self.G_loss_adv_local = -self.completed_local_logit.mean()
            self.G_loss = self.G_loss + self.lambda_adv * (
                self.G_loss_adv + self.G_loss_adv_local
            )

    def forward_D(self):
        # (Pdb) self.completed.shape
        # torch.Size([2, 1, 16, 32, 32])
        # (Pdb) self.completed_local.shape
        # torch.Size([2, 1, 8, 16, 16])
        self.completed_logit, self.completed_local_logit = self.netD(
            self.completed.detach(), self.completed_local.detach()
        )
        self.gt_logit, self.gt_local_logit = self.netD(self.gt, self.gt_local)
        # hinge loss
        self.D_loss_local = (
            nn.ReLU()(1.0 - self.gt_local_logit).mean()
            + nn.ReLU()(1.0 + self.completed_local_logit).mean()
        )
        self.D_loss = (
            nn.ReLU()(1.0 - self.gt_logit).mean()
            + nn.ReLU()(1.0 + self.completed_logit).mean()
        )
        self.D_loss = self.D_loss + self.D_loss_local

    def backward_G(self):
        self.G_loss.backward()

    def backward_D(self):
        self.D_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        self.initVariables()
        self.pred = self.netGM(self.gin)
        self.completed = self.pred * self.mask_01 + self.gt * (1 - self.mask_01)
        if self.opt.mask_type == "rect":
            self.completed_local = self.completed[
                :,
                :,
                self.rect[0] : self.rect[0] + self.rect[1],
                self.rect[2] : self.rect[2] + self.rect[3],
                self.rect[4] : self.rect[4] + self.rect[5],
            ]
        else:
            self.completed_local = self.completed

        if self.opt.pretrain_network is False:
            for i in range(self.opt.D_max_iters):
                self.optimizer_D.zero_grad()
                self.optimizer_G.zero_grad()
                self.forward_D()
                self.backward_D()
                self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.forward_G()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_losses(self):
        l = {
            "G_loss": self.G_loss.item(),
            "G_loss_rec": self.G_loss_reconstruction.item(),
            "G_loss_ae": self.G_loss_ae.item(),
        }
        if self.opt.pretrain_network is False:
            l.update(
                {
                    "G_loss_adv": self.G_loss_adv.item(),
                    "G_loss_adv_local": self.G_loss_adv_local.item(),
                    "D_loss": self.D_loss.item(),
                }
            )
            #   'G_loss_mrf': self.G_loss_mrf.item()})
        return l

    def get_current_visuals(self):
        return {
            "input": self.im_in.cpu().detach().numpy(),
            "gt": self.gt.cpu().detach().numpy(),
            "completed": self.completed.cpu().detach().numpy(),
        }

    def get_current_visuals_tensor(self):
        return {
            "input": self.im_in.cpu().detach(),
            "gt": self.gt.cpu().detach(),
            "completed": self.completed.cpu().detach(),
        }

    def evaluate(self, im_in, mask):
        xin = torch.cat((im_in, mask), 1)
        ret = self.netGM(xin) * mask + im_in * (1 - mask)
        return ret
