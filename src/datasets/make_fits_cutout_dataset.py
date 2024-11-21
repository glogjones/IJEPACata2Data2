# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
from logging import getLogger
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from cata2data import CataData

# Initialize logger and seed
_GLOBAL_SEED = 0
logger = getLogger()

# Function to create a dataset and DataLoader for `.fits` file cutouts
def make_fits_cutout_dataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    cutout_size=224,
    shuffle=True,
    drop_last=True,
    image_path=None,
    catalogue_path=None
):
    """
    Creates a DataLoader for `.fits` file cutouts using CataData.

    Args:
        transform (callable): Transformations to apply to the cutouts.
        batch_size (int): Batch size for the DataLoader.
        collator (callable, optional): Custom collator function for batching.
        pin_mem (bool): Whether to pin memory for faster transfer to GPU.
        num_workers (int): Number of workers for DataLoader.
        world_size (int): Number of distributed processes.
        rank (int): Rank of the current process.
        cutout_size (int): Size of each cutout.
        shuffle (bool): Whether to shuffle the dataset.
        drop_last (bool): Whether to drop the last incomplete batch.
        image_path (str): Path to the FITS image file.
        catalogue_path (str): Path to the catalogue file.

    Returns:
        tuple: Dataset, DataLoader, and DistributedSampler.
    """
    dataset = FitsCutoutDataset(
        catalogue_path=catalogue_path,
        image_path=image_path,
        cutout_size=cutout_size,
        transform=transform
    )
    logger.info('FITS cutout dataset created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('FITS cutout data loader created')

    return dataset, data_loader, dist_sampler


class FitsCutoutDataset(Dataset):
    """
    A Dataset class for `.fits` file cutouts using CataData.
    """
    def __init__(self, catalogue_path, image_path, cutout_size=224, transform=None):
        """
        Initialize the FitsCutoutDataset.

        Args:
            catalogue_path (str): Path to the catalogue file.
            image_path (str): Path to the FITS image file.
            cutout_size (int): Size of each cutout.
            transform (callable, optional): Transformations to apply to the cutouts.
        """
        # Initialize the CataData object
        self.cata_data = CataData(
            catalogue_paths=[catalogue_path],
            image_paths=[image_path],
            cutout_shape=cutout_size,
            field_names=['COSMOS'],  # Single field name for compatibility
            catalogue_kwargs={'format': 'commented_header', 'delimiter': ' '}
        )
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.cata_data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: The transformed cutout image.
        """
        cutout, metadata = self.cata_data[idx]
        if self.transform:
            cutout = self.transform(cutout)
        return cutout


# Function to create a transformation pipeline
def create_transform():
    """
    Create a transformation pipeline for the cutout images.

    Returns:
        callable: Composed transformations.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust mean and std as needed
    ])
    return transform
