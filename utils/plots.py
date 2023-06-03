import os
import random
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import xarray as xr
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .dataloader import FireDataset

# import cv2
# import albumentations as albu

train_on_gpu = torch.cuda.is_available()

def print_features(ds, day: int):
    """plots all features in list for given day"""
    # calculate how many rows are needed
    num_features = len(ds.features)
    rows = num_features // 3 + 1
    cols = 3
    # initialise subplot
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    fig.suptitle(
        f'Normalised features for {ds.ds.time[day].dt.strftime("%B %d, %Y").item()}'
    )
    for i in range(rows):
        for j in range(cols):
            pos = i * 3 + j
            axes[i, j].axis("off")
            # there might be empty subplots at the end
            if pos < num_features:
                ds.ds[ds.features[pos]][day].plot(ax=axes[i, j], add_labels=False)
                axes[i, j].set_title(ds.features[pos])
    plt.show()


def print_target(ds: FireDataset, day: int):
    "plots soft and hard target for given day"
    # initialise subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f'Soft and hard target for {ds.ds.time[day].dt.strftime("%B %d, %Y").item()}'
    )

    target = ds.ds[ds.target].isel(time=29).load().fillna(0)
    target.plot(ax=ax1, add_labels=False)
    ax1.set_title("Soft target")
    target_hard = target.where(target == 0, 1)
    target_hard.plot(ax=ax2, add_labels=False)
    ax2.set_title("Hard target")

    plt.show()


def visualize_image(
    image,
    x_coords=None,
    y_coords=None,
    ax=None,
    return_output=False,
    countries=None,
    **kwargs_imshow,
):
    """
    Plot a single image from a 2D torch.
    Optionally specify the lat/lon coords and the state boarders and it can return the built xarray.DataArray.
    INPUTS:
      image: torch.Tensor with shape = (width, length)
      x_coords: numpy.array
      y_coords: numpy.array
      ax: matplotlib.pyplot.axis
      return_output: bool
      countries: geopandas.GeoDataFrame
    OUTPUTS:
      datarray: xarray.datarray
    """
    # defining coords
    if not isinstance(image, xr.DataArray):
        xc = range(image.shape[0]) if x_coords is None else x_coords
        yc = range(image.shape[1], 0, -1) if y_coords is None else y_coords
        image = xr.DataArray(image, coords={"latitude": yc, "longitude": xc})
    if not ax:
        fig, ax = plt.subplots()
    image.plot.imshow(ax=ax, **kwargs_imshow)
    # plot country boarders in case they are present in the inputs
    if countries is not None:
        countries.boundary.plot(ax=ax, color="grey", lw=0.5)
        ax.set(
            xlim=[min(image.longitude), max(image.longitude)],
            ylim=[min(image.latitude), max(image.latitude)],
        )
    if return_output:
        return image


def visualize_batch_prediction(
    model,
    loader,
    predict_on_gpu=True,
    n_images=5,
    clean_axis=True,
    x_coords=None,
    y_coords=None,
    **kwargs_imshow,
):
    """
    Visualize a number :n_images: of pair which is a comparison between the prediction
    of the :model: and the target image contained in the :loader:.
    INPUTS:
      model - trained model
      loader: torch.DataLoader
      predict_on_gpu: bool - weather or not to predict the images on the gpu.
      x_coords: numpy.array
      y_coords: numpy.array
      clean_axis: bool - weather or not to delete the x/y ticks in the plot.
      kwargs_imshow: additional parameters for matplotlib.pyplot.imshow
    example:
      visualize_batch_prediction(model, valid_loader, n_images = 10, x_coords = tiny_ds.longitude.values, y_coords = tiny_ds.latitude.values, cmap = 'Reds', vmin = 0, vmax = 1)
    """
    fig, axs = plt.subplots(n_images, 2, figsize=(8, 4 * n_images))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    il = iter(loader)
    for it, row_ax in enumerate(axs):
        data, target = next(il)
        if predict_on_gpu:
            data = data.cuda().float()
        output = model(data).detach().cpu().numpy()
        # if a batch of images is extracted take the image with more burned area
        if len(target.shape) == 3:
            indx = (target != 0).sum(axis=[1, 2]).argmax()
            target = target[indx]
            output = output[indx]
        visualize_image(
            output,
            ax=row_ax[0],
            return_output=False,
            countries=countries,
            x_coords=x_coords,
            y_coords=y_coords,
            **kwargs_imshow,
        )
        visualize_image(
            target,
            ax=row_ax[1],
            return_output=False,
            countries=countries,
            x_coords=x_coords,
            y_coords=y_coords,
            **kwargs_imshow,
        )
        if it == 0:
            row_ax[0].set_title("OUTPUT", fontsize=12)
            row_ax[1].set_title("TARGET", fontsize=12)
        if clean_axis:
            [x.axis("off") for x in row_ax]

def visualize_data(
    loader,
    variables_selected,
    variable_to_show="rel_hum",
    n_images=5,
    clean_axis=True,
    x_coords=None,
    y_coords=None,
    **kwargs_imshow,
):
    """
    Visualize a number :n_images: of pair which is a comparison between the prediction
    of the :model: and the target image contained in the :loader:.
    INPUTS:
      model - trained model
      loader: torch.DataLoader
      predict_on_gpu: bool - weather or not to predict the images on the gpu.
      x_coords: numpy.array
      y_coords: numpy.array
      clean_axis: bool - weather or not to delete the x/y ticks in the plot.
      kwargs_imshow: additional parameters for matplotlib.pyplot.imshow
    example:
      visualize_batch_prediction(model, valid_loader, n_images = 10, x_coords = tiny_ds.longitude.values, y_coords = tiny_ds.latitude.values, cmap = 'Reds', vmin = 0, vmax = 1)
    """
    fig, axs = plt.subplots(n_images, 2, figsize=(8, 4 * n_images))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    il = iter(loader)

    dict_var = dict(zip(variables_selected, range(len(variables_selected))))

    for it, row_ax in enumerate(axs):
        data, target = next(il)
        # if a batch of images is extracted take the image with more burned area
        if len(target.shape) == 3:
            indx = (target != 0).sum(axis=[1, 2]).argmax()
            target = target[indx]
            data = data[indx]
        data = data[dict_var[variable_to_show]]

        visualize_image(
            data,
            ax=row_ax[0],
            return_output=False,
            countries=countries,
            x_coords=x_coords,
            y_coords=y_coords,
            **kwargs_imshow,
        )
        visualize_image(
            target,
            ax=row_ax[1],
            return_output=False,
            countries=countries,
            x_coords=x_coords,
            y_coords=y_coords,
            **kwargs_imshow,
        )
        if it == 0:
            row_ax[0].set_title("Data"+" "+variable_to_show+" "+"slice", fontsize=12)
            row_ax[1].set_title("TARGET", fontsize=12)
        if clean_axis:
            [x.axis("off") for x in row_ax]




def print_attributions(
    ds: FireDataset, pred: torch.Tensor, input: torch.Tensor, attributions: torch.Tensor
):
    """
    Plot feature attributions for each input feature in a dataset.
    INPUTS:
      ds: utils.FireDataset
      pred: torch.Tensor with shape = (width, length)
      input: torch.Tensor with shape = (no_features, width, length)
      attributions: torch.Tensor with shape = (no_features, width, length)
    """
    # calculate how many rows are needed
    num_features = len(ds.features)
    rows = len(ds.features)
    cols = 3
    # initialise subplot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    # fig.suptitle('Feature attributions')
    for i in range(num_features):
        feature_name = ds.features[i]
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
        axes[i, 2].axis("off")
        visualize_image(input[i], ax=axes[i, 0])
        axes[i, 0].set_title(f"raw input for {feature_name}")
        visualize_image(attributions[i], ax=axes[i, 1])
        axes[i, 1].set_title(f"attribution for {feature_name}")
        visualize_image(pred, ax=axes[i, 2], cmap="Reds")
        axes[i, 2].set_title(f"prediction")

    fig.tight_layout(pad=1.0)
    plt.show()
    
