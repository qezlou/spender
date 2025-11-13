import glob
import os
import urllib.request
from functools import partial
import h5py

import astropy.io.fits as fits
import astropy.table as aTable
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from ..instrument import Instrument
import ray
import re


class HETDEX(Instrument):
    """HETDEX instrument

    Implements basic parameterization of the HETDEX spectrograph as well as functions
    to download and organize the spectra from the DR16 data archive.
    """

    def __init__(self,wave_obs=None, lsf=None, calibration=None):
        """Create instrument

        Parameters
        ----------
        lsf: :class:`LSF`
            (optional) Line spread function model
        calibration: callable
            (optional) function to calibrate the observed spectrum
        """
        if wave_obs is  None:
            wave_obs = torch.arange(3600, 5301, 2, dtype=torch.float32)
        else:
            wave_obs = wave_obs
        super().__init__(wave_obs, lsf=lsf, calibration=calibration)
    
    def get_data_loader(
        self,
        dir=None,
        file_name=None,
        which=None,
        batch_size=1024,
        shuffle = True,
        get_shotids=False,
        split_ratio=0.98,
        seed=42
    ):
        """Get a dataloader for batches of spectra

        Parameters
        ----------
        dir: string
            Root directory for data storage
        which: ['train', 'valid', 'test'] or None
            Which subset of the spectra to return. If `None`, returns all of them.
        tag: string
            Name to specify which batch files to load
        batch_size: int
            Number of spectra in each batch
        shuffle: bool
            Whether to shuffle the order of the batch files
        shuffle_instance: bool
            Whether to shuffle spectra within each batch
        split_ratio: float
            Fraction of data to use for training (rest is validation)
        seed: int
            Random seed for shuffling data
        Returns
        -------
        :class:`torch.utils.data.DataLoader`
        """
        shards = self.get_shard_paths(dir, file_name)
        if len(shards) > 1:
            num_blocks = 32
        else:
            num_blocks = 1
        dataset = ray.data.read_numpy(shards, override_num_blocks=num_blocks)
        train_ds, valid_ds = dataset.random_shuffle().split_proportionately([1-split_ratio, split_ratio], seed=seed)

        if which == "train":
            return train_ds.iter_torch_batches(batch_size=batch_size, shuffle=shuffle)
        elif which == "valid":
            return valid_ds.iter_torch_batches(batch_size=batch_size, shuffle=shuffle)

    def get_shard_paths(self, dir, file_name):
        """Get list of shard file paths

        Parameters
        ----------
        dir: string
            Root directory for data storage
        file_name: string
            Name to specify which batch files to load

        Returns
        -------
        list of strings
            List of shard file paths
        """
        shard_paths = glob.glob(os.path.join(dir, 'fib_spec', file_name, '*.h5'))
        # Make sure ther are in-order, i.e. newest date-shots are first
        # Files are saved in reverse date order
        tags = [int(re.search(r'c(\d+)', fname).group(1)) for fname in shard_paths]
        shard_paths = [x for _, x in sorted(zip(tags, shard_paths))]
        return shard_paths

    def read_h5_batch(self, shard_path):
        with h5py.File(shard_path, "r") as f:
            spec = f["calfib"][:]
            err = f["calfibe"][:]
            shotids = f["shotids"][:]
            err[err <= 0] = np.inf
            ivar = 1.0 / err**2
            z = np.zeros(spec.shape[0])
            norm = np.ones(spec.shape[0])
            # We don't use the mask here, yet!!
            mask = np.where(err <= 0, 1, 0)
            return {"spec": spec, "ivar": ivar, "mask": mask, "z": z, "shotids": shotids, "norm": norm}
    