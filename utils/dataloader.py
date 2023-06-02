import os
import random
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import xarray as xr
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.notebook import tqdm as ntqdm


def check_for_fire_presence(ds_y, el, min_pixels=1):
    """
    Checks if at least one burned area is present
    """
    rr = dict(latitude=el[0], longitude=el[1])
    mask = (ds_y.isel(rr) > 0).sum(["latitude", "longitude"]).load() >= min_pixels
    return mask.values


def create_list_usable_chunks(ds):
    """
    Uses as input the xarray read from the .zarr file (a dask object)
    and for each one of its chunks returns a Dataframe listing only the chunks that contains fire.
    """
    xx, yy = ds.isel(time=0).chunks
    xinit = 0
    el_tot = []
    nn_cnk = 0
    for xt in ntqdm(xx, desc="check on chuncks with fire"):
        xt += xinit
        yinit = 0
        for yt in yy:
            yt += yinit
            nn_cnk += 1
            el = [slice(xinit, xt), slice(yinit, yt)]
            dd = [
                [nn, *el, rr, nn_cnk]
                for nn, rr in enumerate(check_for_fire_presence(ds, el))
            ]
            el_tot.extend(dd)
            yinit = yt
        xinit = xt
    all_coords = pd.DataFrame(
        el_tot, columns=["time", "latitude", "longitude", "accepted", "chunk_nbr"]
    )
    print(
        f"out of {len(all_coords)} chuncks, {all_coords.accepted.sum()} have at least 1 fire pixel ({all_coords.accepted.sum()/len(all_coords.accepted)*100:.02f}%)"
    )
    return (
        all_coords.replace(False, np.nan)
        .dropna()
        .reset_index()[["time", "latitude", "longitude", "chunk_nbr"]]
    )


class FireDataset(Dataset):
    def __init__(
        self,
        ds: xr.Dataset = None,
        transform_function=None,
        table_mean_std: pd.DataFrame = None,
        time_slice=None,
        preprocessing=None,
        fire_quantiles=None,
        target="fcci_ba",
        inland_map=None,
        delta_time=1,
    ):
        """
        Dataset that contains all the info of teh datacube to be fed to the model
        """
        # save metadata
        features = list(ds.data_vars)
        features.remove(target)
        self.target = target
        self.features = features
        self.delta_time = delta_time
        self.ds = ds.copy()
        # save the data
        self.dx = ds[self.features].shift(time=self.delta_time)
        self.dy = ds[self.target]

        if time_slice is not None:
            self.ds = self.ds.sel(time=time_slice)
            self.dx = self.dx.sel(time=time_slice)
            self.dy = self.dy.sel(time=time_slice)
        # select usable chunks
        self.accepted_slices = create_list_usable_chunks(self.dy)
        # load target data
        self.dy = self.dy.load()

        size_example = self.dy.isel(
            self.accepted_slices[["time", "latitude", "longitude"]].iloc[0].to_dict()
        ).shape
        # load input data into memory
        sol = np.empty(
            (len(self.accepted_slices), len(features), size_example[0], size_example[1])
        )
        idx = 0
        for gg, vv in ntqdm(
            self.accepted_slices.groupby("chunk_nbr"),
            total=len(self.accepted_slices.chunk_nbr.drop_duplicates()),
            desc="loading in memory",
        ):
            sol[idx : idx + len(vv)] = (
                self.dx.isel((vv[["latitude", "longitude"]].iloc[0]).to_dict())
                .isel(time=vv["time"].values)
                .to_array()
                .transpose("time", "variable", "latitude", "longitude")
                .values
            )
            # ds[features].isel((vv[['latitude','longitude']].iloc[0]).to_dict()).isel(time = vv['time'].values).values
            idx += len(vv)
        self.dx = sol

        self.transforms = transform_function
        if table_mean_std is not None:
            self.transforms = transforms.Compose(
                [
                    torch.tensor,
                    # Mean and standard deviation stats used to normalize the input data
                    # to the mean of zero and standard deviation of one.
                    transforms.Normalize(
                        table_mean_std.loc[features, "mean"],
                        table_mean_std.loc[features, "std"],
                    ),
                ]
            )
        self.preprocessing = preprocessing
        self.inland_map = inland_map
        if fire_quantiles is not None:
            self.fire_quantiles = fire_quantiles.loc[target]
        else:
            self.fire_quantiles = None

    def __len__(self):
        return len(self.accepted_slices)

    def __getitem__(self, idx):
        img = self.dx[idx].astype(np.float32)
        if self.transforms:
            img = self.transforms(img)

        if self.fire_quantiles is not None:
            mask = (
                self.dy.isel(self.accepted_slices.iloc[idx, :-1].to_dict())
                .fillna(-1)
                .to_numpy()
            )
            for i, (quantile, quantile_value) in enumerate(
                self.fire_quantiles.items()
            ):
                quantile = float(quantile[0])
                if i == 0:
                    # zeros (not burned area, but on land) should stay the same
                    prev_quantile_value = quantile_value
                    # every value that is higher (or equal to) previous quantile and lower or equal to current quantile receives i (e.g. 1 for first quantile)
                else:
                    mask = np.where(
                        (mask >= prev_quantile_value) & (mask <= quantile_value),
                        quantile,
                        mask,
                    )
                    prev_quantile_value = quantile_value
        else:
            mask = self.dy.isel(self.accepted_slices.iloc[idx, :-1].to_dict()).fillna(0)
            mask = mask.where(mask == 0, 1).to_numpy()
        if self.inland_map is not None:
            mm = self.inland_map.isel(
                self.accepted_slices.iloc[idx][["latitude", "longitude"]].to_dict()
            ).values
            mask = np.where(mm, mask, -2)
        if self.preprocessing:
            pass
        else:
            img = torch.nan_to_num(img, nan=0.0)
        return img, mask


def create_datasets_model(
    ds,
    slice_train,
    slice_valid,
    slice_test,
    table_mean_std=None,
    target="fcci_ba",
    delta_time=1,
    fire_quantiles=None,
    **kwargs_firedaatset,
):
    if slice_test is not None:
        test_dataset = FireDataset(
            ds,
            time_slice=slice_test,
            table_mean_std=table_mean_std,
            delta_time=delta_time,
            fire_quantiles=fire_quantiles,
            target=target,
            **kwargs_firedaatset,
        )
    else:
        test_dataset = None

    if slice_train is not None:
        train_dataset = FireDataset(
            ds,
            time_slice=slice_train,
            table_mean_std=table_mean_std,
            delta_time=delta_time,
            fire_quantiles=fire_quantiles,
            target=target,
            **kwargs_firedaatset,
        )
    else:
        train_dataset = None

    if slice_valid is not None:
        valid_dataset = FireDataset(
            ds,
            time_slice=slice_valid,
            table_mean_std=table_mean_std,
            delta_time=delta_time,
            fire_quantiles=fire_quantiles,
            target=target,
            **kwargs_firedaatset,
        )
    else:
        valid_dataset = None

    return train_dataset, valid_dataset, test_dataset
