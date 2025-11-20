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
        raw = self.load_raw_file(dir,seed=seed, file_name=file_name)
        spec = raw["spec"]
        w = raw["ivar"]  # weight
        z = raw["z"]
        shotids = raw['shotids']
        n = len(spec)
        split = int(split_ratio * n)
        norm = raw['norm']
        if which == "train":
            spec, w, z, shotids = spec[:split], w[:split], z[:split], shotids[:split]
        elif which == "valid":
            spec, w, z, shotids = spec[split:], w[split:], z[split:], shotids[split:]
        
        if get_shotids:
            dataset = torch.utils.data.TensorDataset(spec, w, z, shotids)
        else:
            dataset = torch.utils.data.TensorDataset(spec, w, z)
        torch.manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), norm

    def load_raw_file(self, data_dir, seed, file_name='all_calfibs.h5', normalize=False, frac_each_file=0.1):
        """Load pre-saved HETDEX fiber spectra in h5 format

        Parameters
        ----------
        file_name: string
            Path to local HDF5 file containing the calfib data

        Returns
        -------
        data: dict
            Dictionary with keys 'spec', 'ivar', 'mask', 'z' containing the
            spectra, inverse variance weights, mask, and redshift arrays.
            Redshift is set to zero as it is irrelevant for HETDEX data.
        """
        file_name = glob.glob(os.path.join(data_dir, 'fib_spec', file_name))
        print(f'Number of files found: {len(file_name)}', flush=True)
        if len(file_name) == 0:
            raise FileNotFoundError(f"No file named {file_name} found in {os.path.join(data_dir, 'fib_spec')}")

        for i, fname in enumerate(file_name):
            with h5py.File(fname, 'r') as f:
                ind_rand = np.random.choice(f['calfib'].shape[0], int(frac_each_file * f['calfib'].shape[0]), replace=False)
                ind_rand = np.sort(ind_rand)
                spec_f = torch.from_numpy(f['calfib'][:,65:916][ind_rand])
                calfibe_f = f['calfibe'][:,65:916][ind_rand]
                shotids_f = torch.from_numpy(f['shotids'][:][ind_rand])
                calfibe_f[calfibe_f <= 0] = np.inf  # avoid zero or negative fluxes
                ivar_f = torch.from_numpy(1.0 / calfibe_f**2)
                # For bad pixels, set ivar to zero
                mask_f = torch.from_numpy(np.where(f['calfibe'][:,65:916][ind_rand] <= 0, 1, 0))
                ivar_f[mask_f == 1] = 0.0
                z_f = torch.from_numpy(np.zeros(ind_rand.shape[0]))
                if i == 0:
                    spec = spec_f
                    ivar = ivar_f
                    mask = mask_f
                    z = z_f
                    shotids = shotids_f
                else:
                    spec = torch.cat((spec, spec_f), dim=0)
                    ivar = torch.cat((ivar, ivar_f), dim=0)
                    mask = torch.cat((mask, mask_f), dim=0)
                    z = torch.cat((z, z_f), dim=0)
                    shotids = torch.cat((shotids, shotids_f), dim=0)
        print(f'done loading {spec.shape[0]} spectra', flush=True)
        # Normalize the spectra using the median in the range 4300-5200AA
        sel = (self.wave_obs >= 4300) & (self.wave_obs <= 5200)
        if normalize:
            norm = torch.median(spec[:, sel])
        else:
            norm = torch.ones(spec.shape[1])
        spec = spec  / norm
        ivar = ivar * (norm**2).unsqueeze(0)
        torch.manual_seed(2*seed)
        # Shuffle spectra and associated arrays
        n_spec = spec.shape[0]
        perm = torch.randperm(n_spec)
        spec = spec[perm]
        ivar = ivar[perm]
        mask = mask[perm]
        z = z[perm]
        shotids = shotids[perm]

        return {'spec': spec, 'ivar': ivar, 'mask': mask, 'z': z, 'shotids': shotids, 'norm': norm}
