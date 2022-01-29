import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from typing import Optional
from model.layer import VGG19FeatLayer
from functools import reduce
from monai.networks.layers import GaussianFilter


class ConfidenceDrivenMaskLayer3D(nn.Module):
    def __init__(self, size=65, sigma=1.0 / 40, iters=7):
        super(ConfidenceDrivenMaskLayer3D, self).__init__()
        self.size = size
        self.sigma = sigma
        self.iters = iters
        self.propagationLayer = GaussianFilter(3, sigma)

    def forward(self, mask):
        # here mask 1 indicates missing pixels and 0 indicates the valid pixels
        init = 1 - mask
        mask_confidence = None
        for i in range(self.iters):
            mask_confidence = self.propagationLayer(init)
            mask_confidence = mask_confidence * mask
            init = mask_confidence + (1 - mask)
        return mask_confidence

class ModelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = {}
    
    def get_losses(self):
        return {
            k: round(v, 4)
            for k, v in self.losses.items()
        }

class GeneratorLoss(ModelLoss):
    def __init__(
        self,
        lambda_reconstruction: float,
        lambda_ae: float,
        epsilon=1e-4,
        lambda_adversarial: Optional[float] = None,
    ): 
        super().__init__()
        self.lambda_adversarial = lambda_adversarial
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_ae = lambda_ae
        self.epsilon = epsilon

        self.reconstruction_lossfn = nn.L1Loss()
        self.ae_lossfn = nn.L1Loss()
        self.confidence_mask_fn = ConfidenceDrivenMaskLayer3D()

    def forward(self, generator_output, inputs, discriminator_output=None):
        mask = self.confidence_mask_fn(inputs["mask"])

        reconstruction_loss = self.reconstruction_lossfn(
            generator_output["global"] * inputs["mask"], inputs["gt"].detach() * inputs["mask"]
        )

        # NOTE: recall the mask is one-hot so this effectively divides
        # by size of mask
        reconstruction_loss /= torch.mean(inputs["mask"]) + self.epsilon

        ae_loss = self.ae_lossfn(
            generator_output["prediction"] * (1 - inputs["mask"]),
            inputs["gt"].detach() * (1 - inputs["mask"]),
        )
        ae_loss /= torch.mean(1 - inputs["mask"]) + self.epsilon

        G_loss = self.lambda_reconstruction * reconstruction_loss + self.lambda_ae * ae_loss

        self.losses = {
            "G_loss": G_loss.item(),
            "ae_loss": ae_loss.item(),
            "reconstruction_loss": reconstruction_loss.item()
        }

        if discriminator_output is not None:
            G_loss_adv = -discriminator_output["generator_logit"].mean()
            G_loss_adv_local = -discriminator_output["generator_logit_local"].mean()
            G_loss += self.lambda_adversarial * (G_loss_adv + G_loss_adv_local)
            self.losses = { **self.losses, 
                            "G_loss_adv": G_loss_adv.item(), 
                            "G_loss_adv_local": G_loss_adv_local.item(),}

        return G_loss

class DiscriminatorLoss(ModelLoss):
    def __init__(self):
        super().__init__()

    def forward(self, discriminator_output):
        D_loss_local = (
            torch.nn.ReLU()(1.0 - discriminator_output["gt_logit_local"]).mean()
            + torch.nn.ReLU()(1.0 + discriminator_output["generator_logit_local"]).mean()
        )
        D_loss = (
            torch.nn.ReLU()(1.0 - discriminator_output["gt_logit"]).mean()
            + torch.nn.ReLU()(1.0 + discriminator_output["generator_logit"]).mean()
        )
        D_loss = D_loss + D_loss_local

        self.losses = {
            "D_loss_local": D_loss_local.item(),
            "D_loss": D_loss.item(),
        }

        return D_loss

class WGANLoss(nn.Module):
    def __init__(self):
        super(WGANLoss, self).__init__()

    def __call__(self, input, target):
        d_loss = (input - target).mean()
        g_loss = -input.mean()
        return {"g_loss": g_loss, "d_loss": d_loss}


def gradient_penalty(xin, yout, mask=None):
    gradients = autograd.grad(
        yout,
        xin,
        create_graph=True,
        grad_outputs=torch.ones(yout.size()).cuda(),
        retain_graph=True,
        only_inputs=True,
    )[0]
    if mask is not None:
        gradients = gradients * mask
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def random_interpolate(gt, pred):
    batch_size = gt.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).cuda()
    # alpha = alpha.expand(gt.size()).cuda()
    interpolated = gt * alpha + pred * (1 - alpha)
    return interpolated


class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {"relu3_2": 1.0, "relu4_2": 1.0}
        self.feat_content_layers = {"relu4_2": 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(
            3, patch_size, patch_stride
        )
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i : i + 1, :, :, :]
            gen_feat_i = gen_normalized[i : i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = -(cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(
            rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2
        )[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)

        style_loss_list = [
            self.feat_style_layers[layer]
            * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
            for layer in self.feat_style_layers
        ]
        self.style_loss = (
            reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style
        )

        content_loss_list = [
            self.feat_content_layers[layer]
            * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
            for layer in self.feat_content_layers
        ]
        self.content_loss = (
            reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content
        )

        return self.style_loss + self.content_loss


class StyleLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer, style_layers=None):
        super(StyleLoss, self).__init__()
        self.featlayer = featlayer()
        if style_layers is not None:
            self.feat_style_layers = style_layers
        else:
            self.feat_style_layers = {"relu2_2": 1.0, "relu3_2": 1.0, "relu4_2": 1.0}

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        feats = x.view(b * c, h * w)
        g = torch.mm(feats, feats.t())
        return g.div(b * c * h * w)

    def _l1loss(self, gen, tar):
        return torch.abs(gen - tar).mean()

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [
            self.feat_style_layers[layer]
            * self._l1loss(
                self.gram_matrix(gen_vgg_feats[layer]),
                self.gram_matrix(tar_vgg_feats[layer]),
            )
            for layer in self.feat_style_layers
        ]
        style_loss = reduce(lambda x, y: x + y, style_loss_list)
        return style_loss


class ContentLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer, content_layers=None):
        super(ContentLoss, self).__init__()
        self.featlayer = featlayer()
        if content_layers is not None:
            self.feat_content_layers = content_layers
        else:
            self.feat_content_layers = {"relu4_2": 1.0}

    def _l1loss(self, gen, tar):
        return torch.abs(gen - tar).mean()

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        content_loss_list = [
            self.feat_content_layers[layer]
            * self._l1loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
            for layer in self.feat_content_layers
        ]
        content_loss = reduce(lambda x, y: x + y, content_loss_list)
        return content_loss


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        h_x, w_x = x.size()[2:]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, : h_x - 1, :])
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, : w_x - 1])
        loss = torch.sum(h_tv) + torch.sum(w_tv)
        return loss
