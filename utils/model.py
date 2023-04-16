import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.notebook import tqdm as ntqdm
from torch.nn import MSELoss

# import albumentations as albu
# import xarray as xr
# import cv2
# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt

train_on_gpu = torch.cuda.is_available()


def seed_everything(seed):
    # python seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # numpy seed
    np.random.seed(seed)
    # pytorch seed for all devices (CPU & CUDA)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


####### LOSSES #################
# This loss combines Dice loss with the standard binary cross-entropy (BCE) loss
# that is generally the default for segmentation models.
# Combining the two methods allows for some diversity in the loss,
# while benefitting from the stability of BCE.


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation="sigmoid"):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta**2) * tp + eps) / (
        (1 + beta**2) * tp + beta**2 * fn + fp + eps
    )

    return score


class DiceLoss(nn.Module):
    """Dice Loss"""

    __name__ = "dice_loss"

    def __init__(self, eps=1e-7, activation="sigmoid"):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(
            y_pr,
            y_gt,
            beta=1,
            eps=self.eps,
            threshold=None,
            activation=self.activation,
        )


class BCEDiceLoss(DiceLoss):
    """Standard Binary Cross-Entropy Loss"""

    __name__ = "bce_dice_loss"

    def __init__(
        self, eps=1e-7, activation="sigmoid", lambda_dice=1.0, lambda_bce=1.0, mask=None
    ):
        super().__init__(eps, activation)
        if activation == None:
            self.bce = nn.BCELoss(reduction="mean")
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.mask = mask
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, y_pr, y_gt):
        # calling parent class, i.e. DiceLoss
        dice = super().forward(y_pr, y_gt)
        if self.mask is not None:
            bce = self.bce(y_pr[self.mask], y_gt[self.mask])
        else:
            bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice * dice) + (self.lambda_bce * bce)


def dice_no_threshold(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
):
    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice


##### UNET MODEL ########
# Bulding funtions
class double_conv(nn.Module):
    """Defines 2 subsequent convolutions (conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        # We can use nn.Sequential to combine several subsequent operations
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# UNET Model
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, regression=False):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, False)
        self.up2 = up(512, 128, False)
        self.up3 = up(256, 64, False)
        self.up4 = up(128, 64, False)
        self.outc = outconv(64, n_classes)
        self.regression = regression
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        if self.regression:
            return x.squeeze()
        else:
            return torch.sigmoid(x).squeeze(dim=1)

    def train_model(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        n_epochs: int = 32,
        t_mask=None,
        criterion=MSELoss() if self.regression else BCEDiceLoss(eps=1.0, activation=None, mask=None),
        optimizer=None,
        scheduler=None,
    ):
        if not optimizer:
            optimizer = optim.Adam(self.parameters(), lr=0.005)

        if not scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.2, patience=2, cooldown=2
            )

        train_loss_list = []
        valid_loss_list = []
        dice_score_list = []
        lr_rate_list = []
        valid_loss_min = np.Inf  # track change in validation loss

        for epoch in range(1, n_epochs + 1):
            print(f"epoch: {epoch}")
            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            dice_score = 0.0
            ###################
            # train the model #
            ###################
            self.train()
            bar = ntqdm(train_loader, postfix={"train_loss": 0.0})
            for data, target in bar:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda().float(), target.cuda().float()
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self(data)
                # add 1 dimension to the mask to become [batch, y_dim, x_dim]
                if t_mask is not None:
                    mask_batch = t_mask.unsqueeze(0).expand(output.size())
                    # calculate the batch loss
                    loss = criterion(output[mask_batch], target[mask_batch])
                else:
                    loss = criterion(output, target)
                    # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)
                bar.set_postfix(ordered_dict={"train_loss": loss.item()})
            ######################
            # validate the model #
            ######################
            self.eval()
            del data, target
            with torch.no_grad():
                bar = ntqdm(
                    valid_loader, postfix={"valid_loss": 0.0, "dice_score": 0.0}
                )
                for data, target in bar:
                    # move tensors to GPU if CUDA is available
                    if train_on_gpu:
                        data, target = data.cuda().float(), target.cuda().float()
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = self(data)
                    if t_mask is not None:
                        # add 1 dimension to the mask to become [batch, y_dim, x_dim]
                        mask_batch = t_mask.unsqueeze(0).expand(output.size())
                        # calculate the batch loss
                        loss = criterion(output[mask_batch], target[mask_batch])
                    else:
                        loss = criterion(output, target)
                    # update average validation loss
                    valid_loss += loss.item() * data.size(0)
                    dice_cof = dice_no_threshold(output.cpu(), target.cpu()).item()
                    dice_score += dice_cof * data.size(0)
                    bar.set_postfix(
                        ordered_dict={"valid_loss": loss.item(), "dice_score": dice_cof}
                    )

            # calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)
            dice_score = dice_score / len(valid_loader.dataset)
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            dice_score_list.append(dice_score)
            lr_rate_list.append(
                [param_group["lr"] for param_group in optimizer.param_groups]
            )

            # print training/validation statistics
            print(
                "Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f} Dice Score: {:.6f}".format(
                    epoch, train_loss, valid_loss, dice_score
                )
            )

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                        valid_loss_min, valid_loss
                    )
                )
                torch.save(self.state_dict(), "model_fire.pt")
                valid_loss_min = valid_loss

            scheduler.step(valid_loss)

        return (
            train_loss_list,
            valid_loss_list,
            dice_score_list,
            lr_rate_list,
            valid_loss_min,
        )
